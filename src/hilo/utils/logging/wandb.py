import logging
from omegaconf import OmegaConf
import wandb


def init_wandb(model, cfg):
    wandb_cfg = cfg["logging"].get("wandb", {})
    logger = logging.getLogger(__name__)
    if wandb_cfg.get("enabled", False):
        tags = wandb_cfg.get("tags", [])
        cmd_tags = wandb_cfg.get("extra_tags", [])
        tags = tags + cmd_tags
        if len(tags) == 0:
            tags = None

        wandb.init(
            project=cfg["logging"]["wandb"]["project"],
            entity=cfg["logging"]["wandb"]["entity"],
            mode=cfg["logging"]["wandb"]["mode"],
            name=cfg["logging"]["wandb"]["name"],
            dir=cfg["logging"]["wandb"]["dir"],
            tags=tags,
            config=OmegaConf.to_object(cfg),
        )

        logger.info(f"Initialized wandb with run id: {wandb.run.id}")

        init_wandb_watch(model, wandb_cfg.get("watch", {}))

        return True

    logger.info("Wandb logging is disabled.")
    return False


def init_wandb_watch(mode, watch_cfg):
    logger = logging.getLogger(__name__)
    if watch_cfg.get("enabled", False):
        wandb.watch(mode, log="all", log_freq=watch_cfg.get("frequency", 1000))
        logger.info(
            f"Wandb is watching the model with frequency {watch_cfg.get('frequency', 1000)}."
        )
    else:
        logger.info("Wandb watch is disabled.")
