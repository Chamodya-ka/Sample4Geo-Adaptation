import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroundToAerialCrossAttention(nn.Module):
    """
    Asymmetric cross-attention: ground patches query aerial patches.

    Ground image acts as the source of queries — it asks "which aerial
    patches are relevant to what I see?". Keys and values both come from
    the aerial image, so the output is a ground-shaped summary of only
    the aerial content that matters for the current ground view.

    No positional encoding is injected, so the model must learn patch
    relevance purely from patch content similarity across domains.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Separate projections: Q from ground, K/V from aerial.
        # Using separate linear layers (not a fused QKV) because Q and K/V
        # come from different domains — tied weights would be wrong here.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform for Q/K (dot-product attention), Kaiming for V/out
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        ground: torch.Tensor,   # (B, N_g, D)
        aerial: torch.Tensor,   # (B, N_a, D)
        aerial_key_padding_mask: torch.Tensor | None = None,  # (B, N_a) bool
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ground: Ground-level patch embeddings.
            aerial: Aerial patch embeddings (larger spatial coverage).
            aerial_key_padding_mask: Optional bool mask — True where aerial
                patches should be ignored (e.g. padding). Shape (B, N_a).
            return_attn_weights: If True, also return the attention map
                (B, H, N_g, N_a) — useful for visualisation / analysis.

        Returns:
            attended: (B, N_g, D) — ground queries enriched with the
                      relevant aerial context.
            attn_weights (optional): (B, H, N_g, N_a)
        """
        B, N_g, D = ground.shape
        B2, N_a, D2 = aerial.shape
        assert B == B2 and D == D2, "Batch size and embed_dim must match"

        H, d_k = self.num_heads, self.head_dim

        # Project each domain into Q / K / V spaces
        Q = self.q_proj(ground)   # (B, N_g, D)
        K = self.k_proj(aerial)   # (B, N_a, D)
        V = self.v_proj(aerial)   # (B, N_a, D)

        # Reshape to multi-head format: (B, H, N, d_k)
        def reshape_heads(x: torch.Tensor, seq_len: int) -> torch.Tensor:
            return x.view(B, seq_len, H, d_k).transpose(1, 2)  # (B, H, N, d_k)

        Q = reshape_heads(Q, N_g)  # (B, H, N_g, d_k)
        K = reshape_heads(K, N_a)  # (B, H, N_a, d_k)
        V = reshape_heads(V, N_a)  # (B, H, N_a, d_k)

        # Scaled dot-product attention: (B, H, N_g, N_a)
        # Each ground patch attends over ALL aerial patches simultaneously.
        # The model learns which aerial patches to upweight purely from the
        # content of the embeddings — no positional bias is added.
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Mask padded aerial patches before softmax
        if aerial_key_padding_mask is not None:
            # Expand to (B, 1, 1, N_a) so it broadcasts over heads and N_g
            mask = aerial_key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, H, N_g, N_a)
        attn_weights = self.attn_drop(attn_weights)

        # Aggregate values: (B, H, N_g, d_k)
        attended = torch.matmul(attn_weights, V)

        # Merge heads back: (B, N_g, D)
        attended = attended.transpose(1, 2).contiguous().view(B, N_g, D)
        attended = self.out_proj(attended)

        if return_attn_weights:
            return attended, attn_weights
        return attended


class CrossAttentionBlock(nn.Module):
    """
    Full transformer block wrapping GroundToAerialCrossAttention.

    Follows the Pre-LN convention (LayerNorm before sublayer) which is more
    stable to train and doesn't require a warm-up schedule for the LR.

    Stack N of these if you want a deeper selection mechanism.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4

        self.norm_ground = nn.LayerNorm(embed_dim)
        self.norm_aerial = nn.LayerNorm(embed_dim)

        self.cross_attn = GroundToAerialCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)


    def forward(
        self,
        ground: torch.Tensor,   # (B, N_g, D)
        aerial: torch.Tensor,   # (B, N_a, D)
        aerial_key_padding_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN cross-attention with residual
        B = ground.shape[0]
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        ground = torch.cat([cls, ground], dim=1)  # (B, N_g+1, D)
        g_norm = self.norm_ground(ground)
        a_norm = self.norm_aerial(aerial)

        if return_attn_weights:
            ca_out, attn_w = self.cross_attn(
                g_norm, a_norm,
                aerial_key_padding_mask=aerial_key_padding_mask,
                return_attn_weights=True,
            )
        else:
            ca_out = self.cross_attn(
                g_norm, a_norm,
                aerial_key_padding_mask=aerial_key_padding_mask,
            )

        x = ground + ca_out  # residual

        # Pre-LN FFN with residual
        x = x + self.ffn(self.norm_ffn(x))

        aerial_descriptor = x[:, 0, :]  # (B, D) — the [CLS] token now contains the fused info
        if return_attn_weights:
            return aerial_descriptor, attn_w
        return aerial_descriptor