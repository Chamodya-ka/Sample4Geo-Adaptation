import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train_adapted(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    if train_config.train_adapter_only:
        for param in model.parameters():
            param.requires_grad = True
        #freeze base model parameters
        for param in model.base_model.parameters():
            param.requires_grad = False

    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    c=0
    # for loop over one epoch
    for query, reference, ids in bar:
        c+=1
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                features1, features2, stage3_features1, stage3_features2 = model(grd_img=query, aerial_img=reference)
                #stage3_features1 = stage3_features1.reshape(train_config.batch_size,-1, 1024)
                #stage3_features2 = stage3_features2.reshape(train_config.batch_size,-1, 1024)
                # Adding OT module
                # print(ot_features1.shape, features2.shape)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    # weighted_aerial_features = low_fow_adapter_transformer(stage3_features1, stage3_features2)
                    infoNCE_loss = loss_function(features1, features2, model.base_model.logit_scale.exp())
                    #ot_loss, _ = cvft_module(stage3_features1, stage3_features2, torch.eye(features1.size(0)).to(train_config.device))
                    
                else:
                    # weighted_aerial_features = low_fow_adapter_transformer(stage3_features1, stage3_features2)
                    infoNCE_loss = loss_function(features1, features2, model.base_model.logit_scale.exp()) 
                    #ot_loss, _ = cvft_module(stage3_features1, stage3_features2, torch.eye(features1.size(0)).to(train_config.device))
                total_loss = infoNCE_loss # + alpha * ot_loss
                # if c%25==0:
                #     print("InfoNCE Loss: ", infoNCE_loss.item(), "OT Loss: ", ot_loss.item())
                losses.update(total_loss.item())
                
                  
            scaler.scale(total_loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            # bfloat no scaler
            with torch.amp.autocast(device_type=torch.device(train_config.device).type, dtype=torch.bfloat16):
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)

                # Forward pass
                features1, features2, stage3_features1, stage3_features2 = model(grd_img=query, aerial_img=reference)
                # Adding OT module
                # ot_features1 = cvft_module(features1, features2, torch.eye(features1.size(0)).to(train_config.device))[0]
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    infonce_loss = loss_function(features1, features2, model.base_model.logit_scale.exp())
                    # ot_loss, _ = cvft_module(features1, features2, torch.eye(features1.size(0)).to(train_config.device))
                else:
                    infonce_loss = loss_function(features1, features2, model.base_model.logit_scale.exp()) 
                    # ot_loss, _ = cvft_module(features1, features2, torch.eye(features1.size(0)).to(train_config.device))
                losses.update(infonce_loss.item())
            loss = infonce_loss.float()
            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
            
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(total_loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader, query_features_stage3=None):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    query_features_stage3_list = []
    ids_list = []
    with torch.no_grad():
        if query_features_stage3 is not None:
            # query_features_stage3 = query_features_stage3.to(train_config.device)
            for data, query_feat in zip(bar, query_features_stage3):
                img, ids = data
                ids_list.append(ids)
                with torch.amp.autocast(device_type=torch.device(train_config.device).type, dtype=torch.bfloat16):
                    img = img.to(train_config.device)
                    query_feat = query_feat.to(train_config.device)
                    img_feature, _ = model(aerial_img=img, query_features=query_feat)
                    if train_config.normalize_features:
                        img_feature = F.normalize(img_feature, dim=-1)
                img_features_list.append(img_feature.to(torch.float32))
        else:
            for img, ids in bar:
                ids_list.append(ids)
                with torch.amp.autocast(device_type=torch.device(train_config.device).type, dtype=torch.bfloat16):
                    img = img.to(train_config.device)
                    # we are calculating query features
                    img_feature, query_features_stage3_ = model(grd_img=img)
                    query_features_stage3_list.append(query_features_stage3_)
                    # normalize is calculated in fp32
                    if train_config.normalize_features:
                        img_feature = F.normalize(img_feature, dim=-1)
                # save features in fp32 for sim calculation
                img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list, query_features_stage3_list