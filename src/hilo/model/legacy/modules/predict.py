import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class PredictionBlock(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(PredictionBlock, self).__init__()
        self.linear = nn.Linear(input_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MLPModule(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.1, batch_size=8, max_seq_len=20, num_sensors=5):
        super(MLPModule, self).__init__()

        self.mlp = nn.Sequential()
        self.output_size = output_size
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        # last_channel = in_channels

        pre_norm_rearrange = Rearrange("(B S O) C -> B C (S O)", B=batch_size, O=max_seq_len, S=num_sensors)
        after_norm_rearrange = Rearrange("B C (S O) -> (B S O) C", B=batch_size, O=max_seq_len, S=num_sensors)

        for i in range(len(layer_sizes) - 1):
            self.mlp.add_module(
                f"Linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            if i < len(layer_sizes) - 2:
                # self.mlp.add_module(
                #     f"pre_norm_rearrange_{i}", pre_norm_rearrange
                # )
                self.mlp.add_module(
                    f"norm_{i}", nn.BatchNorm1d(num_features=layer_sizes[i + 1])
                )
                # self.mlp.add_module(
                #     f"after_norm_rearrange_{i}", after_norm_rearrange
                # )
                self.mlp.add_module(f"nonlin_{i}", nn.ReLU())
                # self.mlp.add_module(f'dropout_{i}', CustomDropout(dropout_prob))

    def forward(self, x):
        x = self.mlp(x)
        return x


class CustomDropout(nn.Module):
    def __init__(self, dropout_prob):
        super(CustomDropout, self).__init__()
        self.dropout = nn.Dropout1d(dropout_prob)

    def forward(self, x):
        x = x.transpose(1, 0)  # Swap dimensions (l, c) -> (c, l)
        x = self.dropout(x)
        x = x.transpose(1, 0)  # Swap dimensions back (c, l) -> (l, c)
        return x
