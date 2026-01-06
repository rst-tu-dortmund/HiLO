import torch
from torch import nn


class Normalization(nn.Module):
    """
    Selective normalization helper.

    Config keys:
      - enabled: bool
      - dim: int (feature/channel dim, default 1)
      - norm_indices: optional list/tuple/1D-tensor of indices to normalize
      - center: optional list/1D-array/tensor (either full feature length or length == len(norm_indices))
      - scale: optional list/1D-array/tensor (same rules as center)

    Behavior:
      - forward(x): returns x with selected indices normalized; non-selected channels unchanged.
      - denormalize(x, feature_idcs=None, norm_idcs=None):
          * If feature_idcs is provided, x is assumed to be a full-feature tensor and the channels in
            feature_idcs will be denormalized in-place and returned.
          * If norm_idcs is provided, x contains only the normalized channels (in order) and will be denormalized.
    """
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.dim = int(cfg.get("dim", 1))

        idxs = cfg.get("norm_indices", None)
        if idxs is None:
            self.norm_idcs = None
        else:
            self.norm_idcs = torch.tensor(idxs, dtype=torch.long)

        # register center/scale as buffers so they exist as attributes and can be mutated in tests
        center_cfg = cfg.get("center", None)
        scale_cfg = cfg.get("scale", None)
        if center_cfg is not None:
            center_t = torch.tensor(center_cfg, dtype=torch.float32)
        else:
            center_t = torch.tensor([], dtype=torch.float32)
        if scale_cfg is not None:
            scale_t = torch.tensor(scale_cfg, dtype=torch.float32)
        else:
            scale_t = torch.tensor([], dtype=torch.float32)

        # buffers keep device/dtype when .to() is called on module
        self.register_buffer("center", center_t)
        self.register_buffer("scale", scale_t)

    def _resolve_param(self, param_buf: torch.Tensor, x_sel: torch.Tensor, x_full: torch.Tensor):
        """Return a tensor for param matching x_sel.size(self.dim). Accepts param either for selection or full-length."""
        if param_buf is None or param_buf.numel() == 0:
            # default center=0, scale=1
            if param_buf is self.center:
                return torch.zeros(x_sel.size(self.dim), dtype=torch.float32, device=x_sel.device)
            else:
                return torch.ones(x_sel.size(self.dim), dtype=torch.float32, device=x_sel.device)

        p = param_buf.to(x_sel.device, dtype=torch.float32)
        # if p matches selected length already, use directly
        if p.numel() == x_sel.size(self.dim):
            return p
        # if p matches full feature length and norm indices are present, slice
        if self.norm_idcs is not None and p.numel() == x_full.size(self.dim):
            return p[self.norm_idcs].to(x_sel.device)
        # otherwise shape mismatch
        raise ValueError(
            f"Provided parameter length ({p.numel()}) does not match selected feature length ({x_sel.size(self.dim)}) "
            f"nor full feature length ({x_full.size(self.dim)})."
        )

    def _reshape_for_broadcast(self, vec: torch.Tensor, x: torch.Tensor):
        # create shape for broadcasting along dim
        shape = [1] * x.dim()
        shape[self.dim] = vec.numel()
        return vec.view(*shape)

    def forward(self, x: torch.Tensor, mask=None):
        if not self.enabled:
            return x.clone()

        # select portion to normalize
        if self.norm_idcs is None:
            x_sel = x
        else:
            x_sel = torch.index_select(x, self.dim, self.norm_idcs.to(x.device))

        # resolve center/scale to match x_sel (center/scale buffers may be full-length or selection-length)
        center = self._resolve_param(self.center, x_sel, x)
        scale = self._resolve_param(self.scale, x_sel, x)

        # ensure device/dtype and broadcastable shape
        center = center.to(x.device, dtype=x.dtype)
        scale = scale.to(x.device, dtype=x.dtype)
        center_b = self._reshape_for_broadcast(center, x_sel)
        scale_b = self._reshape_for_broadcast(scale, x_sel)

        x_sel_norm = (x_sel - center_b) / scale_b

        if self.norm_idcs is None:
            return x_sel_norm

        # put normalized values back into full tensor
        x_out = x.clone()
        x_out.index_copy_(self.dim, self.norm_idcs.to(x.device), x_sel_norm)
        return x_out

    def denormalize(self, x: torch.Tensor, feature_idcs=None, norm_idcs=None):
        """
        Denormalize values.

        - If feature_idcs is provided: x is full-feature tensor and feature_idcs selects channels to denorm.
        - Else if norm_idcs is provided: x contains only the normalized channels (in order) and will be denormalized.
        """
        if not self.enabled:
            return x.clone()

        # Case A: x is full-feature tensor, feature_idcs selects channels to denorm
        if feature_idcs is not None:
            feature_idcs = feature_idcs.to(x.device)
            x_sel = torch.index_select(x, self.dim, feature_idcs)
            center = self._resolve_param(self.center, x_sel, x)
            scale = self._resolve_param(self.scale, x_sel, x)
            center_b = self._reshape_for_broadcast(center.to(x.device, dtype=x.dtype), x_sel)
            scale_b = self._reshape_for_broadcast(scale.to(x.device, dtype=x.dtype), x_sel)
            x_sel_den = x_sel * scale_b + center_b
            x_out = x.clone()
            x_out.index_copy_(self.dim, feature_idcs, x_sel_den)
            return x_out

        # Case B: x contains only normalized channels; norm_idcs tells where to place them
        if norm_idcs is None:
            raise ValueError("When denormalizing a tensor of selected channels, 'norm_idcs' must be provided.")

        norm_idcs = norm_idcs.to(x.device)
        # determine center/scale for these norm_idcs relative to full (try matching lengths)
        if self.center.numel() == norm_idcs.numel():
            center_sel = self.center.to(x.device, dtype=x.dtype)
            scale_sel = self.scale.to(x.device, dtype=x.dtype)
        else:
            # assume provided as full-length -> slice
            center_sel = self.center.to(x.device, dtype=x.dtype)[norm_idcs]
            scale_sel = self.scale.to(x.device, dtype=x.dtype)[norm_idcs]

        center_b = self._reshape_for_broadcast(center_sel, x)
        scale_b = self._reshape_for_broadcast(scale_sel, x)
        x_den = x * scale_b + center_b

        return x_den
