
from sample4geo.transformer_low_fov_adapter import CrossAttentionBlock
import torch
import torch.nn as nn
import torchvision.models as models

class AdaptedTimmModel(nn.Module):
    def __init__(self, pretrained_model, freeze_base=False):
        super(AdaptedTimmModel, self).__init__()
        # 1. Load the pretrained base (excluding its original classifier/head)
        self.base_model = pretrained_model
        
        # Optionally freeze base model weights
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            
        # 2. Define your new custom module
        self.adapter = self.low_fov_adapter_transformer = CrossAttentionBlock(embed_dim=1024, num_heads=8, ffn_dim=4096, dropout=0.1)
#

    def forward(self, grd_img, aerial_img=None):
        if aerial_img is not None:
            features1, features2, stage3_features1, stage3_features2 = self.base_model(grd_img, aerial_img)
            stage3_features1 = stage3_features1.reshape(stage3_features1.shape[0],-1, 1024)
            stage3_features2 = stage3_features2.reshape(stage3_features2.shape[0],-1, 1024)
            weighted_features2 = self.adapter(stage3_features1, stage3_features2)
            return features1, weighted_features2, stage3_features1, stage3_features2
        else:
            features1 = self.base_model(grd_img)
            #weighted_features2 = self.adapter(features1, features2)
            return features1#, weighted_features2, stage3_features1, stage3_features2
       


      #  self.low_fov_adapter_transformer = CrossAttentionBlock(embed_dim=1024, num_heads=8, ffn_dim=4096, dropout=0.1)
#