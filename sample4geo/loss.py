import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn


class InfoNCE(nn.Module):

    def __init__(
        self, loss_function, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T

        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(
            len(logits_per_image1), dtype=torch.long, device=self.device
        )

        loss = (
            self.loss_function(logits_per_image1, labels)
            + self.loss_function(logits_per_image2, labels)
        ) / 2

        return loss


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss (Wang et al., CVPR 2019) for conditional embeddings.

    Expects matched pairs: feats1[i] and feats2[i] are a positive pair
    (e.g. cond_embed1[i] and cond_embed2[i] from the two symmetric CA branches).
    All cross-index pairs within the batch are treated as negatives.

    Loss per anchor i:
        L_i = (1/α) * log[1 + Σ_{k∈P_i} exp(-α(s_ik − λ))]
            + (1/β) * log[1 + Σ_{k∈N_i} exp( β(s_ik − λ))]

    Reference: https://arxiv.org/abs/1904.06627
    """

    def __init__(self, alpha: float = 2.0, beta: float = 50.0, lam: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats1: (B, d) — conditional embeddings from branch 1 (img1 feat maps | img2 query)
            feats2: (B, d) — conditional embeddings from branch 2 (img2 feat maps | img1 query)
        Returns:
            Scalar MS loss.
        """
        B = feats1.shape[0]
        feats1 = F.normalize(feats1, dim=-1)
        feats2 = F.normalize(feats2, dim=-1)

        # Pool both sets: indices 0..B-1 are from feats1, B..2B-1 from feats2.
        # feats1[i] and feats2[i] share the same location label i.
        all_feats = torch.cat([feats1, feats2], dim=0)  # (2B, d)
        labels = torch.arange(B, device=feats1.device).repeat(2)  # [0..B-1, 0..B-1]

        # Full cosine similarity matrix
        sim = all_feats @ all_feats.T  # (2B, 2B)

        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (2B, 2B) bool
        neg_mask = ~pos_mask
        pos_mask.fill_diagonal_(False)  # exclude self

        loss = feats1.new_zeros(1)
        count = 0
        for i in range(2 * B):
            pos_sim = sim[i][pos_mask[i]]
            neg_sim = sim[i][neg_mask[i]]

            if pos_sim.numel() == 0 or neg_sim.numel() == 0:
                continue

            pos_loss = (1.0 / self.alpha) * torch.log(
                1.0 + torch.sum(torch.exp(-self.alpha * (pos_sim - self.lam)))
            )
            neg_loss = (1.0 / self.beta) * torch.log(
                1.0 + torch.sum(torch.exp(self.beta * (neg_sim - self.lam)))
            )
            loss = loss + pos_loss + neg_loss
            count += 1

        return loss / max(count, 1)
