from .optimal_match_loss import OptimalMatchLoss
from.legacy.loss import LegacyMatchingLoss

all_losses = {
    "optimalmatchloss": OptimalMatchLoss,
    "legacymatchingloss": LegacyMatchingLoss,
}


def build_loss(cfg):
    loss_type = cfg.get("name", "optimalmatchloss").lower()
    if loss_type in all_losses:
        return all_losses[loss_type](cfg)
    raise ValueError(f"Unknown loss type: {loss_type}")
