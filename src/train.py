import torch
import hydra
from tqdm import tqdm
import copy
import logging

from hilo.data import build_dataloader
from hilo.model import build_model
from hilo.loss import build_loss

from hilo.training.routine import train_one_epoch
from hilo.utils.training.setup import set_fixed_seed
from hilo.utils.training.lr_schedulers import initialize_lr_scheduler
from hilo.utils.training.saving import save_topk_checkpoints
from hilo.evaluation.routine import evaluate_model, should_evaluate
from hilo.utils.logging.wandb import init_wandb

import omegaconf_resolver

omegaconf_resolver.register()


@hydra.main(config_path="hilo/config", config_name="train", version_base="1.3")
def train(cfg):
    logging.basicConfig(
        level=cfg["logging"]["level"],
        format=cfg["logging"]["format"],
        datefmt=cfg["logging"]["datefmt"],
    )
    set_fixed_seed(cfg["training"]["seed"], cfg["training"].get("deterministic", True))

    train_loader = build_dataloader(cfg["data"], split="train")

    val_data_cfg = copy.deepcopy(cfg["data"])
    val_data_cfg["shuffle"] = False
    val_loader = build_dataloader(val_data_cfg, split="val")
    
    cfg["data"]["num_samples"]["train"] = len(train_loader.dataset)
    cfg["data"]["num_samples"]["val"] = len(val_loader.dataset)

    model = build_model(cfg["model"])
    model.to(cfg["device"])

    loss = build_loss(cfg["training"]["loss"])
    loss.to(cfg["device"])

    optimizer = torch.optim.AdamW(
        model.get_parameter_groups(
            weight_decay=cfg["training"]["optimizer"].get("weight_decay", 0.01),
            bias_norm_decay=cfg["training"]["optimizer"].get("bias_norm_decay", True),
        ),
        lr=cfg["training"].get("learning_rate", 1e-3),
    )

    lr_scheduler = initialize_lr_scheduler(
        cfg["training"]["scheduler"], optimizer, len(train_loader)
    )

    topk_checkpoints = None

    use_wandb = init_wandb(model, cfg)
    epochs = cfg["training"]["epochs"]
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_one_epoch(
            model,
            train_loader,
            loss,
            optimizer,
            epoch,
            lr_scheduler=lr_scheduler,
            use_wandb=use_wandb,
            visualize_cfg=cfg["training"].get("visualization", None),
        )

        if should_evaluate(cfg["evaluation"], epoch, epochs):
            metrics = evaluate_model(
                model,
                val_loader,
                epoch,
                cfg["evaluation"]["metrics"],
                use_wandb=use_wandb,
                visualize_cfg=cfg["evaluation"].get("visualization", None),
            )
            topk_checkpoints = save_topk_checkpoints(
                cfg["logging"]["checkpoints"],
                model,
                optimizer,
                lr_scheduler,
                epoch,
                metrics,
                topk_checkpoints,
            )


if __name__ == "__main__":
    train()
