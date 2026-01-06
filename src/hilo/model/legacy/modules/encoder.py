import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from hilo.model.legacy.modules.attention import MultiheadAttention
from torch import Tensor
from typing import Optional
import copy


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_prob, batch_first=False
        )
        # self.self_attention = MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout1d(dropout_prob)
        self.num_heads = num_heads

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pos, key_padding_mask):
        # x: (seq_length, batch_size, embed_dim)
        # self-attention
        query = key = self.with_pos_embed(x, pos)
        # attn_mask = torch.matmul(key_padding_mask.float().transpose(1,0), key_padding_mask.float()).bool()
        attn_mask = key_padding_mask.unsqueeze(1) * key_padding_mask.unsqueeze(2)
        attn_mask = torch.repeat_interleave(attn_mask, self.num_heads, dim=0)
        attn_output, attn_scores = self.self_attention(
            query, key, x, key_padding_mask=None, attn_mask=attn_mask
        )
        attn_output = self.dropout(attn_output)
        x = self.layer_norm1(x + attn_output)

        # feed-forward
        ff_output = self.feedforward(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm2(x + ff_output)

        return x


class Encoder(nn.Module):
    def __init__(
        self, num_layers, embed_dim, num_heads, feedforward_dim, dropout_prob=0.1
    ):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, num_heads, feedforward_dim, dropout_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, pos, key_padding_mask):
        # x: (batch_size, seq_length, embed_dim)
        for layer in self.layers:
            x = layer(x, pos, key_padding_mask)
        return x


# class TransformerEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu"):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout1d(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout1d(dropout)
#         self.dropout2 = nn.Dropout1d(dropout)

#         self.activation = _get_activation_fn(activation)

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward(self,
#                      src,
#                      src_mask: Optional[Tensor] = None,
#                      src_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None):
#         q = k = self.with_pos_embed(src, pos)
#         src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2.permute(1,2,0)).permute(2,0,1)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src)).permute(1,2,0)).permute(2,0,1))
#         src = src + self.dropout2(src2.permute(1,2,0)).permute(2,0,1)
#         src = self.norm2(src)
#         return src


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", use_original=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout1d(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.lnc2ncl = Rearrange("L N C -> N C L")
        self.ncl2lnc = Rearrange("N C L -> L N C")
        self.dropout1 = nn.Dropout1d(dropout)
        self.dropout2 = nn.Dropout1d(dropout)

        if use_original:
            self.lnc2ncl = nn.Identity()
            self.ncl2lnc = nn.Identity()
            self.dropout = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.ncl2lnc(self.dropout1(self.lnc2ncl(src2)))
        src = self.norm1(src.permute(1, 0, 2)).permute(1, 0, 2)
        src2 = self.linear2(self.ncl2lnc(self.dropout(self.lnc2ncl(self.activation(self.linear1(src))))))
        src = src + self.ncl2lnc(self.dropout2(self.lnc2ncl(src2)))
        src = self.norm2(src.permute(1, 0, 2)).permute(1, 0, 2)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SensorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, use_original=False):
        super(SensorEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            input_dim, num_heads, hidden_dim, dropout=0.1, activation="relu", use_original=use_original
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, sensor_data):
        # Encode using the transformer encoder
        encoder_output = self.transformer_encoder(sensor_data)
        return encoder_output
