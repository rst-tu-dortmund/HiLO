import torch
from torch.nn import Module


class Normalization(Module):
    def __init__(self, cfg):
        super(Normalization, self).__init__()
        self.register_buffer("center", torch.tensor(cfg["center"]))
        self.register_buffer("scale", torch.tensor(cfg["scale"]))
        norm_idcs = cfg.get("norm_indices", None)
        if norm_idcs is not None:
            self.register_buffer("norm_idcs", torch.tensor(norm_idcs, dtype=torch.long))
        else:
            self.norm_idcs = None

        self.dim = cfg.get("dim", -1)

        self.track_running_stats = cfg.get(
            "track_running_stats", "none"
        )  # "none", "center", "scale", "both"
        self.momentum = cfg.get("momentum", 0.1)
        self.track_center = False
        self.track_scale = False
        if (
            self.track_running_stats.lower() is None
            or self.track_running_stats.lower() == "none"
        ):
            self.track_running_stats = None
        elif self.track_running_stats.lower() == "center":
            self.track_center = True
        elif self.track_running_stats.lower() == "scale":
            self.track_scale = True
        elif self.track_running_stats.lower() == "both":
            self.track_center = True
            self.track_scale = True
        else:
            raise ValueError(
                f"Unknown track_running_stats option: {self.track_running_stats}"
            )

    def _adjust_shape(self, x):
        # adjust shape for broadcasting
        shape = [1] * x.dim()
        shape[self.dim] = -1
        center = self.center.view(shape)
        scale = self.scale.view(shape)

        return center, scale

    def track_stats(self, x, mask=None):
        if self.track_running_stats is None or not self.training:
            return

        # calculate running stats along all other dims except self.dim
        dim = self.dim % x.dim()
        reduce_dims = tuple(d for d in range(x.dim()) if d != dim)

        if mask is not None and mask.any():
            x_valid = x[mask]
            reduce_dims = 0
        else:
            x_valid = x

        if self.track_center:
            batch_center = x_valid.mean(dim=reduce_dims, keepdim=True).squeeze()
            self.center.mul_(1 - self.momentum).add_(batch_center * self.momentum)

        if self.track_scale:
            batch_var = x_valid.var(dim=reduce_dims, unbiased=False, keepdim=True)
            batch_scale = batch_var.sqrt().squeeze()
            self.scale.mul_(1 - self.momentum).add_(batch_scale * self.momentum)
            self.scale.clamp_(min=1e-6)

    def forward(self, x, mask=None):
        if self.norm_idcs is None:
            x_sel = x
        else:
            x_sel = torch.index_select(x, self.dim, self.norm_idcs)

        self.track_stats(x_sel, mask)
        center, scale = self._adjust_shape(x_sel)

        if self.norm_idcs is None:
            return (x_sel - center) / scale

        x_sel_norm = (x_sel - center) / scale

        x_norm = torch.index_copy(x, self.dim, self.norm_idcs, x_sel_norm)

        return x_norm

    def denormalize(self, x, feature_idcs=None, norm_idcs=None):
        if feature_idcs is not None:
            x_ = torch.index_select(x, self.dim, feature_idcs)
        else:
            x_ = x

        center, scale = self._adjust_shape(x_)

        if norm_idcs is None:
            x_denorm = x_ * scale + center
            x_denorm_full = (
                torch.index_copy(x, self.dim, feature_idcs, x_denorm)
                if feature_idcs is not None
                else x_denorm
            )
            return x_denorm_full

        scale_ = torch.index_select(scale, self.dim, norm_idcs)
        center_ = torch.index_select(center, self.dim, norm_idcs)

        x_denorm = x_ * scale_ + center_
        x_denorm_full = (
            torch.index_copy(x, self.dim, feature_idcs, x_denorm)
            if feature_idcs is not None
            else x_denorm
        )
        return x_denorm_full
