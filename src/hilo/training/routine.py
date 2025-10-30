import json
from tqdm import tqdm
import wandb
import logging
from hilo.utils.data.management import move_data_to_device
from hilo.utils.visualization.controls import should_visualize
from hilo.utils.visualization.bev import plot_sample

def train_one_epoch(model, train_loader, loss, optimizer, epoch, lr_scheduler=None, use_wandb=False, visualize_cfg=None):
    # TODO: implement logging with wandb if use_wandb is True
    model.train()
    logger = logging.getLogger(__name__)
    device = next(model.parameters()).device
    epoch_loss = None
    
    num_batches = len(train_loader)
    pbar_batches = tqdm(train_loader, desc="Batches", position=1, leave=False, total=num_batches)

    for batch_idx, batch in enumerate(pbar_batches):
        global_training_step = epoch * num_batches + batch_idx
        batch_device = move_data_to_device(batch, device)
        outputs = model(batch_device["data"], batch_device["mask"])

        losses, assignment_cost, association_results = loss(outputs, batch_device["gt_data"], batch_device["gt_mask"])

        losses["total_loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch_loss is None:
            epoch_loss = {k: v.item() for k, v in losses.items()}
        else:
            epoch_loss = {k: epoch_loss[k] + losses[k].item() for k in losses}
            
        pbar_batches.set_description(f"Batches. Epoch avg: {epoch_loss['total_loss'] / (batch_idx + 1):.4f}")
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        fig = None
        if should_visualize(visualize_cfg, batch_idx):
            fig = plot_sample(outputs,
                              batch,
                              class_names=visualize_cfg.get("class_names", None),
                                filter_background=visualize_cfg.get("filter_background", True)
                              )

        if use_wandb:
            log_dict = {f"train/{k}": v.item() for k, v in losses.items()}
            log_dict["lr"] = lr_scheduler.get_current_lr() if lr_scheduler is not None else optimizer.param_groups[0]['lr']
            log_dict["train/global_step"] = global_training_step
            log_dict["train/batch_idx"] = batch_idx
            
            if fig is not None:
                log_dict["train/visualization"] = fig
            
            wandb.log(log_dict)

    avg_dict = {
        f"avg/train/{k}": v / (batch_idx + 1) for k, v in epoch_loss.items()
    }

    logger.info(f"Epoch {epoch} training losses:")
    logger.info(json.dumps(avg_dict, indent=4))
    
    if use_wandb:
        avg_dict["epoch"] = epoch
        wandb.log(avg_dict)

    if lr_scheduler is not None:
        lr_scheduler.epoch()