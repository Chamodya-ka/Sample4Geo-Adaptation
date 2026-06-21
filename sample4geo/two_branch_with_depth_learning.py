import torch
import timm
import numpy as np
import torch.nn as nn

"""
Depth auxiliary learning utilities for cross-view geolocalization.

These let the ground encoder learn depth-aware representations during
training, while remaining a plain image encoder at inference -- no depth
map, no Depth Anything V2, needed at test time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAuxHead(nn.Module):
    """Lightweight depth-regression head for auxiliary supervision.

    Accepts either:
      - CNN feature maps: (B, C, H, W)
      - ViT token sequences: (B, N, C), optionally with prefix tokens
        (cls / register tokens) that get stripped before reshaping to a grid.

    Predicts a single-channel relative depth map. This head only exists
    during training -- drop it entirely at inference.
    """

    def __init__(self, in_dim, mid_dim=128, num_prefix_tokens=0, upsample_factor=4):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.upsample_factor = upsample_factor
        self.decoder = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 3, padding=1),
            nn.GroupNorm(8, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim // 2, 3, padding=1),
            nn.GroupNorm(8, mid_dim // 2),
            nn.GELU(),
            nn.Conv2d(mid_dim // 2, 1, 1),
        )

    def _to_spatial(self, feat, grid_size=None):
        if feat.dim() == 4:
            return feat  # already (B, C, H, W)

        if self.num_prefix_tokens > 0:
            feat = feat[:, self.num_prefix_tokens:, :]

        b, n, c = feat.shape
        if grid_size is None:
            h = w = int(n ** 0.5)
            if h * w != n:
                raise ValueError(
                    f"Token grid of {n} tokens isn't square -- pass grid_size explicitly."
                )
        else:
            h, w = grid_size

        return feat.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, feat, grid_size=None):
        x = self._to_spatial(feat, grid_size)
        depth = self.decoder(x)
        depth = F.relu(depth)
        depth = F.interpolate(
            depth, (224, 308), mode="bilinear", align_corners=True
        )
        # depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.squeeze(1)  # (B, H', W')


def silog_loss(pred, target, mask=None, eps=1e-6):
    """Scale-invariant log loss. Use this instead of plain L1/L2 -- Depth
    Anything V2 outputs *relative* depth, so an absolute-error loss would
    fight the encoder over a scale that was never meaningful to begin with."""
    if mask is None:
        mask = target > eps
    pred = pred[mask].clamp(min=eps)
    target = target[mask].clamp(min=eps)
    d = torch.log(pred) - torch.log(target)
    return (d ** 2).mean() - 0.5 * (d.mean() ** 2)


def gradient_matching_loss(pred, target, mask=None):
    """Multi-scale edge-aware term. Optional -- pushes the encoder to match
    depth *discontinuities* (object/scene boundaries) rather than just
    smooth values, which tends to transfer better to the retrieval task."""
    loss = pred.new_tensor(0.0)
    p0 = pred.unsqueeze(1)
    t0 = target.unsqueeze(1)
    for scale in (1, 2, 4):
        p = F.avg_pool2d(p0, scale) if scale > 1 else p0
        t = F.avg_pool2d(t0, scale) if scale > 1 else t0
        dp_x = p[:, :, :, 1:] - p[:, :, :, :-1]
        dt_x = t[:, :, :, 1:] - t[:, :, :, :-1]
        dp_y = p[:, :, 1:, :] - p[:, :, :-1, :]
        dt_y = t[:, :, 1:, :] - t[:, :, :-1, :]
        loss = loss + (dp_x - dt_x).abs().mean() + (dp_y - dt_y).abs().mean()
    return loss

class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 use_depth_aux=False):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        self.use_depth_aux = use_depth_aux
        self.is_vit = False
        if self.is_vit:
            self.grd_branch = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
            self.aer_branch = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.grd_branch = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.aer_branch = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
 
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.depth_head = None
        if self.use_depth_aux:
            feat_dim = self.grd_branch.num_features
            num_prefix = getattr(self.grd_branch, "num_prefix_tokens", 1 if self.is_vit else 0)
            self.depth_head = DepthAuxHead(feat_dim, num_prefix_tokens=num_prefix)
 
    def get_config(self):
        data_config = timm.data.resolve_model_data_config(self.grd_branch)
        return data_config
 
    def set_grad_checkpointing(self, enable=True):
        self.grd_branch.set_grad_checkpointing(enable)
        self.aer_branch.set_grad_checkpointing(enable)

    def _extract(self, model, img):
        """Return (feat_maps, embedding) for one image in a single forward pass."""
        feat_maps = model.forward_features(img)  # (B, C, H, W) or (B, N, C)
        embedding = model.forward_head(feat_maps)  # (B, d)
        return feat_maps, embedding
    
    def _grid_size(self, model):
        """Patch grid (h, w) for vit token reshaping, if available."""
        patch_embed = getattr(model, "patch_embed", None)
        return getattr(patch_embed, "grid_size", None) if patch_embed is not None else None
 

    def forward(self, img1, img2, predict_depth=False):
        "img1: always query, img2: always reference"
        if img1 is not None and img2 is not None:
       
            feature_maps1, image_features1 = self._extract(self.grd_branch, img1)     
            image_features2 = self.aer_branch(img2)
            if predict_depth and self.depth_head is not None:
                pred_depth = self.depth_head(feature_maps1, grid_size=self._grid_size(self.grd_branch))
                return image_features1, image_features2, pred_depth
            return image_features1, image_features2            
              
        elif img1 is not None:
            # only query image
            feature_maps1, image_features1 = self._extract(self.grd_branch, img1)     
            if predict_depth and self.depth_head is not None:
                pred_depth = self.depth_head(feature_maps1, grid_size=self._grid_size(self.grd_branch))
                return image_features1, pred_depth
            return image_features1
        else:
            # only reference image, no query image
            image_features2 = self.aer_branch(img2)
            return image_features2