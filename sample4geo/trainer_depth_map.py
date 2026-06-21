import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
from sample4geo.two_branch_with_depth_learning import silog_loss, gradient_matching_loss

def train(
    train_config,
    model,
    dataloader,
    loss_function,
    optimizer,
    scheduler=None,
    scaler=None,
    use_depth_aux=True,
):

    # set model train mode
    model.train()

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
    c = 0
    # for loop over one epoch
    for query, reference, pseudo_depth, ids in bar:
        c += 1
        if scaler:
            with autocast():

                # data (batches) to device
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                pseudo_depth = pseudo_depth.to(train_config.device)
                # Forward pass — TimmModel_CA returns 4 values during training
                # (embed1, embed2, cond_embed1, cond_embed2), 2 values at eval.
                out = model(query, reference, predict_depth = True)
                features1, features2, depth = out

                logit_scale = (
                    model.module.logit_scale.exp()
                    if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1
                    else model.logit_scale.exp()
                )
                infoNCE_loss = loss_function(features1, features2, logit_scale)

                # MS loss is applied to the *conditional* embeddings (paper §3)
                if use_depth_aux:
                    scale_inv_loss = silog_loss(depth, pseudo_depth)
                    grad_match_loss =gradient_matching_loss(depth, pseudo_depth)

                total_loss =infoNCE_loss +  train_config.depth_loss_weight * scale_inv_loss + train_config.grad_loss_weight * grad_match_loss
                losses.update(total_loss.item())

            scaler.scale(total_loss).backward()

            # Gradient clipping
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), train_config.clip_grad
                )

            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if (
                train_config.scheduler == "polynomial"
                or train_config.scheduler == "cosine"
                or train_config.scheduler == "constant"
            ):
                scheduler.step()

        else:
            # bfloat no scaler
            with torch.amp.autocast(
                device_type=torch.device(train_config.device).type, dtype=torch.bfloat16
            ):
                # data (batches) to device
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)

                # Forward pass — TimmModel_CA returns 4 values during training.
                out = model(img1=query, img2=reference, predict_depth=True)
                features1, features2, depth = out
            
                logit_scale = (
                    model.module.logit_scale.exp()
                    if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1
                    else model.logit_scale.exp()
                )
                infonce_loss = loss_function(features1, features2, logit_scale)
                scale_inv_loss = silog_loss(depth, pseudo_depth)
                grad_match_loss =gradient_matching_loss(depth, pseudo_depth)

                total_loss = infoNCE_loss +  train_config.depth_loss_weight * scale_inv_loss + train_config.grad_loss_weight * grad_match_loss

                losses.update(total_loss.item())
            loss = total_loss.float()
            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), train_config.clip_grad
                )

            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if (
                train_config.scheduler == "polynomial"
                or train_config.scheduler == "cosine"
                or train_config.scheduler == "constant"
            ):
                scheduler.step()

        if train_config.verbose:

            monitor = {
                "loss": "{:.4f}".format(total_loss.item()),
                "loss_avg": "{:.4f}".format(losses.avg),
                "lr": "{:.6f}".format(optimizer.param_groups[0]["lr"]),
                "depth_losses": "{:.4f}".format(scale_inv_loss + grad_match_loss)
            }

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader, is_query=True):

    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    with torch.no_grad():

        for img, ids in bar:

            ids_list.append(ids)

            with torch.amp.autocast(
                device_type=torch.device(train_config.device).type, dtype=torch.bfloat16
            ):

                img = img.to(train_config.device)
                if is_query:
                    img_feature = model(img, None)
                else:
                    img_feature = model(None, img)

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

    return img_features, ids_list
