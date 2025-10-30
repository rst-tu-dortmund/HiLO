import os
import torch
import numpy as np


def save_topk_checkpoints(
    checkpoint_cfg,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    metrics,
    topk_checkpoints,
):
    if not checkpoint_cfg["enabled"]:
        return

    if topk_checkpoints is None:
        topk_checkpoints = {}
        # first call, current model is the best
        os.makedirs(checkpoint_cfg["dir"], exist_ok=True)
        for metric in checkpoint_cfg["metrics"]:
            topk_checkpoints[metric] = {
                "epochs": [epoch],
                "metrics": np.array(metrics[metric]),
                "paths": [f"{checkpoint_cfg['dir']}/{metric}_{epoch}.pth"],
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                topk_checkpoints[metric]["paths"][-1],
            )
            print(
                f"Saved first checkpoint for {metrics[metric]:.4f} {metric} at epoch {epoch} to {topk_checkpoints[metric]['paths'][-1]}"
            )
    else:
        for metric in checkpoint_cfg["metrics"]:
            current_metric = metrics[metric]

            if len(topk_checkpoints[metric]["epochs"]) < checkpoint_cfg["topk"]:
                # save new checkpoint
                topk_checkpoints[metric]["epochs"].append(epoch)
                topk_checkpoints[metric]["metrics"] = np.append(
                    topk_checkpoints[metric]["metrics"], current_metric
                )
                path = f"{checkpoint_cfg['dir']}/{metric}_{epoch}.pth"
                topk_checkpoints[metric]["paths"].append(path)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    path,
                )
                print(
                    f"Saved new checkpoint for {current_metric:.4f} {metric} at epoch {epoch} to {path}"
                )

            else:
                worst_metric = np.min(topk_checkpoints[metric]["metrics"])
                if current_metric > worst_metric:
                    # replace worst checkpoint
                    worst_idx = np.argmin(topk_checkpoints[metric]["metrics"])
                    topk_checkpoints[metric]["epochs"][worst_idx] = epoch
                    topk_checkpoints[metric]["metrics"][worst_idx] = current_metric
                    path = f"{checkpoint_cfg['dir']}/{metric}_{epoch}.pth"
                    topk_checkpoints[metric]["paths"][worst_idx] = path
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "metrics": metrics,
                        },
                        path,
                    )
                    print(
                        f"Saved new checkpoint for {current_metric:.4f} {metric} at epoch {epoch} to {path}"
                    )

    return topk_checkpoints
