import torch
import torch.nn as nn
import einops


class SinusoidalEmbedding(nn.Module):
    """
    see: https://github.com/PRBonn/ir-mcl/blob/aab8c91b37e59f94442d8eddfa2f920ab1ecc4c5/nof/networks/models.py#L11
    Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)

    :param in_channels: number of input channels (2 for both xyz and direction)
    :param N_freq: number of scale of embeddings
    :param logscale: whether to use log scale (default: True)

    """

    def __init__(self, in_channels, N_freq, logscale=True, **kwargs):
        super(SinusoidalEmbedding, self).__init__()
        self.N_freq = N_freq
        self.in_channels = in_channels

        self.funcs = [torch.sin, torch.cos]

        if logscale:
            freq_bands = 2 ** torch.linspace(0, N_freq - 1, N_freq)
        else:
            freq_bands = torch.linspace(1, 2 ** (N_freq - 1), N_freq)

        self.register_buffer("freq_bands", freq_bands)

    # @profiler_context
    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)

        :param x: (B, self.in_channels)
        :return out: (B, self.N_freq * self.in_channels * len(self.funcs))
        """
        x = x[..., None]
        out = [x]
        prod = self.freq_bands * x

        for func in self.funcs:
            out.append(func(prod))

        out = einops.rearrange(torch.cat(out, -1), "... C F -> ... (C F)")
        return out
