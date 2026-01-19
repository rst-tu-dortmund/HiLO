import torch
import hydra
from tqdm import tqdm
import copy
import logging
import os
import json
import numpy as np

from hilo.data import build_dataloader
from hilo.model import build_model
from hilo.evaluation.routine import evaluate_model
from hilo.utils.training.setup import set_fixed_seed
from hilo.utils.logging.wandb import init_wandb

import omegaconf_resolver

omegaconf_resolver.register()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@hydra.main(config_path="hilo/config", config_name="eval", version_base="1.3")
def evaluate(cfg):
    logging.basicConfig(
        level=cfg["logging"]["level"],
        format=cfg["logging"]["format"],
        datefmt=cfg["logging"]["datefmt"],
    )
    logger = logging.getLogger("eval")

    # Set seed
    set_fixed_seed(cfg.get("seed", 42), True)

    # Build validation loader
    if "data" not in cfg:
        raise ValueError("Data config is missing")

    val_data_cfg = copy.deepcopy(cfg["data"])
    val_data_cfg["shuffle"] = False

    logger.info("Building dataloader...")
    val_loader = build_dataloader(val_data_cfg, split=cfg.get("split", "val"))
    logger.info(f"Dataloader built with {len(val_loader)} batches.")

    # Build model
    logger.info("Building model...")
    model = build_model(cfg["model"])
    model.to(cfg["device"])

    # Load checkpoint
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "Please specify a checkpoint path in the config via 'checkpoint=/path/to/ckpt.pth'"
        )

    # resolve path (hydra might change cwd)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=cfg["device"])

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    msg = model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded checkpoint with message: {msg}")

    # Init wandb
    use_wandb = init_wandb(model, cfg)

    epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0

    # Evaluate
    logger.info("Starting evaluation...")
    metrics = evaluate_model(
        model,
        val_loader,
        epoch,
        cfg["evaluation"]["metrics"],
        use_wandb=use_wandb,
        visualize_cfg=cfg["evaluation"].get("visualization", None),
    )

    logger.info("Evaluation results:")
    print(json.dumps(metrics, indent=4, cls=NumpyEncoder))


if __name__ == "__main__":
    evaluate()
