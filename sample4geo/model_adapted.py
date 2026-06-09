from sample4geo.transformer_low_fov_adapter import CrossAttentionBlock
import torch
import timm
import numpy as np
import torch.nn as nn


class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        # get the model output from later model.stages_3.blocks.2.conv_dw
        features_storage = {}
        target_layer = self.model.stages[3].blocks[-1].conv_dw
        def hook_fn(module, input, output):
            # Store the intermediate activation tensor
            features_storage['stage3_output'] = output.detach()
        hook_handle = target_layer.register_forward_hook(hook_fn)

        if img2 is not None:
       
            image_features1 = self.model(img1)     
            stage3_features1 = features_storage['stage3_output']
            image_features2 = self.model(img2)
            stage3_features2 = features_storage['stage3_output']
            hook_handle.remove()
#            weighted_aerial_features = self.low_fov_adapter_transformer(stage3_features1, stage3_features2)
            return image_features1, image_features2, stage3_features1, stage3_features2            
              
        else:
            image_features = self.model(img1)
            stage3_features = features_storage['stage3_output']
            hook_handle.remove()
            return image_features, stage3_features
