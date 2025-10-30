from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn


class OneCycle(nn.Module):
    def __init__(self, optimizer, max_lr, warmup_epochs, max_epochs, steps_per_epoch):
        super().__init__()
        self.lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max_epochs * steps_per_epoch,
            pct_start=warmup_epochs / max_epochs,
        )

    def step(self):
        self.lr_scheduler.step()

    def epoch(self):  # OneCycleLR does iteration-based scheduling only
        pass

    def get_current_lr(self):
        return self.lr_scheduler.get_last_lr()[0]

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)


def initialize_lr_scheduler(lr_scheduler_cfg, optimizer, num_iterations_per_epoch):
    if (
        lr_scheduler_cfg is None
        or lr_scheduler_cfg.get("type", "none").lower() == "none"
    ):
        return None

    lr_scheduler_type = lr_scheduler_cfg.get("type", "onecycle").lower()
    if lr_scheduler_type in ["onecycle", "one_cycle", "onecyclelr", "one_cycle_lr"]:
        max_epochs = lr_scheduler_cfg["training_epochs"]
        warmup_epochs = lr_scheduler_cfg.get("warmup_epochs", max_epochs // 3)
        max_lr = lr_scheduler_cfg["max_lr"]
        return OneCycle(
            optimizer, max_lr, warmup_epochs, max_epochs, num_iterations_per_epoch
        )
    else:
        raise NotImplementedError(
            f"LR scheduler type '{lr_scheduler_type}' is not implemented yet."
        )
