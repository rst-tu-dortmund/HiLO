import json
from tqdm import tqdm
import torch
import wandb
import logging
from hilo.evaluation.metrics.calculator import MetricCalculator
from hilo.utils.data.management import move_data_to_device

from hilo.utils.visualization.controls import should_visualize
from hilo.utils.visualization.bev import plot_sample


def evaluate_model(
    model, val_loader, epoch, metrics_cfg, use_wandb=False, visualize_cfg=None
):
    model.eval()
    logger = logging.getLogger(__name__)
    device = next(model.parameters()).device

    metric_calculator = MetricCalculator(metrics_cfg)

    num_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(val_loader, desc=f"Evaluating Epoch {epoch}")
        ):
            global_eval_step = epoch * num_batches + batch_idx

            batch_device = move_data_to_device(batch, device)

            outputs = model(batch_device["data"], batch_device["mask"])

            batch_metrics = metric_calculator.batch(outputs, batch_device)

            fig = None
            if should_visualize(visualize_cfg, batch_idx):
                fig = plot_sample(
                    outputs,
                    batch,
                    class_names=visualize_cfg.get("class_names", None),
                    filter_background=visualize_cfg.get("filter_background", True),
                )

            if use_wandb:
                log_dict = {
                    f"eval/{k}": v
                    for k, v in batch_metrics.items()
                    if not "map_raw" in k
                }
                log_dict["eval/global_step"] = global_eval_step
                log_dict["eval/batch_idx"] = batch_idx

                if fig is not None:
                    log_dict["visualization/eval/sample"] = fig

                wandb.log(log_dict)

        epoch_metrics = metric_calculator.epoch()
        logger.info(f"Epoch {epoch} evaluation metrics:")
        logger.info(json.dumps(epoch_metrics, indent=4))

        if use_wandb:
            log_dict = {f"avg/eval/{k}": v for k, v in epoch_metrics.items()}
            log_dict["epoch"] = epoch

            wandb.log(log_dict)

    return epoch_metrics


def should_evaluate(cfg, epoch, epochs):
    return (epoch + 1) % cfg.get("interval", 1) == 0 or epoch + 1 >= cfg.get(
        "eval_after_epoch", epochs
    )
