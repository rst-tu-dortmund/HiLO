import torch
from torch.nn import Module
from torch.nn import functional as F


class FocalLoss(Module):
    def __init__(self, cfg):
        super(FocalLoss, self).__init__()
        self.gamma = cfg.get("gamma", 2.0)
        self.alpha = cfg.get("alpha", 1.0)
        self.reduction = cfg.get("reduction", "mean")

        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        cls_is_obj = targets < inputs.shape[-1] - 1  # last dimension is no-object class

        ce_unweighted = self.ce(inputs, targets.to(torch.long))
        pt = torch.exp(-ce_unweighted)

        focal_weight = (
            self.alpha * cls_is_obj + (1 - self.alpha) * (1 - cls_is_obj)
        ) * (1 - pt).pow(self.gamma)

        loss = ce_unweighted * focal_weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
