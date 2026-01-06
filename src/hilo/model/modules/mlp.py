from torch import nn
from torch.nn import Module
from collections import OrderedDict


class InstanceNormWrapper(nn.Module):
    def __init__(self, num_features):
        super(InstanceNormWrapper, self).__init__()
        self.inst_norm = nn.InstanceNorm1d(num_features)

    def forward(self, x):
        # x shape: ((B * N), C) -> (C, L) for InstanceNorm1d

        x_ = x.permute(1, 0)
        x_norm = self.inst_norm(x_)

        return x_norm.permute(1, 0)


class MLP(Module):
    def __init__(self, cfg):
        """Accepts inputs of shape B x C"""
        super(MLP, self).__init__()
        use_he_init = cfg.get("use_he_init", False)

        in_channels = cfg["in_channels"]
        hidden_channels = cfg["hidden_channels"]
        out_channels = cfg.get("out_channels", None)
        dropout = cfg.get("dropout", 0.0)
        bias = cfg.get("bias", True)

        norm_layer = nn.Identity
        if "norm" in cfg:
            norm_layer_name = cfg["norm"]["name"].lower()
            if norm_layer_name in ["batch", "batchnorm", "batchnorm1d", "bn"]:
                norm_layer = nn.BatchNorm1d
            elif norm_layer_name in ["layer", "layernorm", "ln"]:
                norm_layer = nn.LayerNorm
            elif norm_layer_name in ["group", "groupnorm", "gn"]:
                norm_layer = lambda num_channels: nn.GroupNorm(
                    cfg["norm"]["num_groups"], num_channels
                )
            elif norm_layer_name in ["instance", "instancenorm", "instancenorm1d", "in"]:
                norm_layer = InstanceNormWrapper
            else:
                raise NotImplementedError(
                    f"Normalization layer '{norm_layer_name}' not implemented yet."
                )

        activation_layer = nn.Identity
        if "activation" in cfg:
            activation_name = cfg["activation"]["name"].lower()
            if activation_name in ["relu"]:
                activation_layer = nn.ReLU
            elif activation_name in ["leakyrelu", "leaky_relu", "lrelu"]:
                activation_layer = lambda: nn.LeakyReLU(
                    cfg["activation"].get("negative_slope", 0.01)
                )
            elif activation_name in ["gelu"]:
                activation_layer = nn.GELU
            elif activation_name in ["silu", "swish"]:
                activation_layer = nn.SiLU
            else:
                raise NotImplementedError(
                    f"Activation layer '{activation_name}' not implemented yet."
                )

        if cfg.get("is_intermediate", False) and out_channels is not None:
            hidden_channels.append(out_channels)
            out_channels = None

        layers = []

        in_channels_ = in_channels
        for i in range(len(hidden_channels)):
            lin = nn.Linear(in_channels_, hidden_channels[i], bias=bias)
            if use_he_init:
                nn.init.kaiming_uniform_(lin.weight, mode="fan_in", nonlinearity="relu")
                if bias:
                    nn.init.zeros_(lin.bias)
            layers.append((f"lin_{i}", lin))
            layers.append((f"norm_{i}", norm_layer(hidden_channels[i])))
            layers.append((f"activation_{i}", activation_layer()))
            layers.append((f"drop_{i}", nn.Dropout1d(dropout)))

            in_channels_ = hidden_channels[i]

        if out_channels is not None:
            out_lin = nn.Linear(hidden_channels[-1], out_channels, bias=bias)

            if use_he_init:
                nn.init.kaiming_uniform_(
                    out_lin.weight, mode="fan_in", nonlinearity="relu"
                )
                if bias:
                    nn.init.zeros_(out_lin.bias)

            layers.append(
                (
                    f"out_lin",
                    out_lin,
                )
            )

        self.mlp = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.mlp(x)
