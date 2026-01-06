import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of zeros for positional encodings
        pe = torch.zeros(max_seq_len, embedding_dim)
        # Generate positions from 0 to max_seq_len - 1
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # Compute the div_term for positional encodings
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        # Apply sine positional encodings for even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine positional encodings for odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension to positional encodings
        # pe = pe.unsqueeze(0)
        # Register the positional encodings as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # bs, _, _, _ = x.shape
        # Broadcast the positional encoding to match the batch size of the input tensor
        pe = self.pe.to(x.device)
        # Add positional encodings to input tensor x
        x = x + pe.requires_grad_(False)
        return x

    def getPos(self, x):
        noObj, bs, _ = x.shape
        pe = self.pe.to(x.device)
        # Broadcast the positional encoding to match the batch size of the input tensor
        pe = pe.unsqueeze(1).repeat(1, bs, 1)
        return pe.requires_grad_(False)


class ObjectPositionEmbeddingSine(nn.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, object_parameters, mask=None):
        # object_parameters should be a tensor of shape (batch_size, num_objects, num_features)
        batch_size, num_objects, num_features = object_parameters.size()

        # If mask is not provided, create a mask assuming all objects are valid
        if mask is None:
            mask = torch.ones(
                batch_size,
                num_objects,
                dtype=torch.float32,
                device=object_parameters.device,
            )

        # Create a mask to identify valid objects
        not_mask = mask

        if self.normalize:
            eps = 1e-6
            object_parameters_min = object_parameters.min(dim=1, keepdim=True).values
            object_parameters_max = object_parameters.max(dim=1, keepdim=True).values
            object_parameters_normalized = (
                (object_parameters - object_parameters_min)
                / (object_parameters_max - object_parameters_min + eps)
                * self.scale
            )
        else:
            object_parameters_normalized = object_parameters

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=object_parameters.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Create positional embeddings for object parameters
        pos = object_parameters_normalized[:, :, :] / dim_t
        pos = torch.stack(
            (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        # Apply the mask to set positional embeddings for padding objects to zero
        pos *= not_mask[:, :, None].float()

        return pos
