import math

import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionConditionalEmbedding(nn.Module):
    """
    Single cross-attention block implementing the conditional embedding from
    Kotovenko et al., "Cross-Image-Attention for Conditional Embeddings in DML" (CVPR 2023).

    Given two images I_i and I_j, this module computes:
        φ(E(I_i) | φ(E(I_j))) = CA(φ(E(I_j)), E(I_i), E(I_i))

    Where:
        Q = linear_q(φ(E(I_j)))  in R^(1 × d)   – global embedding of the *other* image
        K = linear_k(E(I_i))     in R^(T × d)   – spatial / sequence features of *this* image
        V = linear_v(E(I_i))     in R^(T × d)   – same source as K

    The output is a d-dimensional conditional embedding of I_i that attends
    to the spatial regions most relevant for a subsequent comparison to I_j.
    Only used during training; inference uses the standard unconditional embedding.
    """

    def __init__(
        self, feat_dim: int, embed_dim: int, num_heads: int = 8, dropout: float = 0.1
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q comes from the global embedding (d-dim); K/V from spatial feature maps (feat_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(feat_dim, embed_dim)
        self.v_proj = nn.Linear(feat_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Pre-LayerNorm (more stable than post-LN)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(feat_dim)

        self.attn_drop = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self, query_embed: torch.Tensor, kv_feature_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_embed:     (B, d)          – φ(E(I_j)), global embedding of the *other* image
            kv_feature_maps: (B, C, H, W) or (B, N, C)
                             – E(I_i), backbone feature maps of *this* image
        Returns:
            cond_embed: (B, d) – φ(E(I_i) | φ(E(I_j))), conditional embedding
        """
        B = query_embed.shape[0]

        # Bring feature maps to sequence form (B, T, C)
        if kv_feature_maps.dim() == 4:  # CNN spatial output: (B, C, H, W)
            _, C, H, W = kv_feature_maps.shape
            kv = kv_feature_maps.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:  # ViT sequence output: (B, N, C)
            kv = kv_feature_maps

        T = kv.shape[1]
        H_n, d_k = self.num_heads, self.head_dim

        # Pre-LN
        q_in = self.norm_q(query_embed)  # (B, d)
        kv_in = self.norm_kv(kv)  # (B, T, C)

        # Linear projections → multi-head layout (B, H_n, seq, d_k)
        Q = self.q_proj(q_in).view(B, 1, H_n, d_k).transpose(1, 2)  # (B, H_n, 1, d_k)
        K = self.k_proj(kv_in).view(B, T, H_n, d_k).transpose(1, 2)  # (B, H_n, T, d_k)
        V = self.v_proj(kv_in).view(B, T, H_n, d_k).transpose(1, 2)  # (B, H_n, T, d_k)

        # Scaled dot-product: single query attends over all T spatial tokens
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H_n, 1, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)  # (B, H_n, 1, d_k)
        # Merge heads: squeeze seq-dim → (B, H_n, d_k) → (B, d)
        out = out.squeeze(2).contiguous().view(B, self.embed_dim)

        return self.out_proj(out)  # (B, d)


class TimmModel_CA(nn.Module):

    def __init__(
        self, model_name, pretrained=True, img_size=383, ca_num_heads=8, ca_dropout=0.1
    ):

        super(TimmModel_CA, self).__init__()

        self.img_size = img_size

        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
        else:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0
            )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ── Probe feature-map and embedding dimensions (CPU, no grad) ──────────
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat_maps = self.model.forward_features(dummy)  # (1, C, H, W) or (1, N, C)
            embed = self.model.forward_head(feat_maps)  # (1, d)

            # feat_dim: channel dim for CNNs, feature dim for ViT sequences
            if feat_maps.dim() == 4:  # (B, C, H, W)
                self.feat_dim = feat_maps.shape[1]
            else:  # (B, N, C) or (B, C)
                self.feat_dim = feat_maps.shape[-1]

            self.embed_dim = embed.shape[-1]

        # ── Two symmetric cross-attention branches ───────────────────────────
        # ca_branch1: Q = embed(img2),  K/V = feat_maps(img1)
        #             → conditional embedding of img1 given img2
        # ca_branch2: Q = embed(img1),  K/V = feat_maps(img2)   [symmetric]
        #             → conditional embedding of img2 given img1
        self.ca_branch1 = CrossAttentionConditionalEmbedding(
            feat_dim=self.feat_dim,
            embed_dim=self.embed_dim,
            num_heads=ca_num_heads,
            dropout=ca_dropout,
        )
        self.ca_branch2 = CrossAttentionConditionalEmbedding(
            feat_dim=self.feat_dim,
            embed_dim=self.embed_dim,
            num_heads=ca_num_heads,
            dropout=ca_dropout,
        )

    def get_config(
        self,
    ):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def _extract(self, img):
        """Return (feat_maps, embedding) for one image in a single forward pass."""
        feat_maps = self.model.forward_features(img)  # (B, C, H, W) or (B, N, C)
        embedding = self.model.forward_head(feat_maps)  # (B, d)
        return feat_maps, embedding

    def forward(self, img1, img2=None):

        if img2 is not None:
            feat1, embed1 = self._extract(img1)
            feat2, embed2 = self._extract(img2)

            if self.training:
                # ── Training path ────────────────────────────────────────────
                # Branch 1: feature maps of img1 as K/V, embedding of img2 as Q
                #   cond_embed1 ≈ φ(E(img1) | φ(E(img2)))
                cond_embed1 = self.ca_branch1(embed2, feat1)

                # Branch 2 (symmetric): feature maps of img2 as K/V, embedding of img1 as Q
                #   cond_embed2 ≈ φ(E(img2) | φ(E(img1)))
                cond_embed2 = self.ca_branch2(embed1, feat2)

                # Returns unconditional embeddings (for InfoNCE / base loss)
                # and conditional embeddings (for MultiSimilarityLoss).
                return embed1, embed2, cond_embed1, cond_embed2

            else:
                # ── Inference path ───────────────────────────────────────────
                # Use the unconditional embeddings improved by CA training.
                # No extra parameters or compute at test time.
                return embed1, embed2

        else:
            # Single-image encoding (gallery / query embedding at eval time)
            _, embedding = self._extract(img1)
            return embedding
