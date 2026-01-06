import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import copy


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_prob, batch_first=False
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.enc_dec_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_prob, batch_first=False
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout1d(dropout_prob)
        self.num_heads = num_heads

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        x,
        encoded_withPos,
        encoder_output,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # x: (seq_length, batch_size, embed_dim)
        # encoder_output: (seq_length, batch_size, embed_dim)

        # self-attention
        query = key = self.with_pos_embed(x, query_pos)
        self_attn_output, self_attn_scores = self.self_attention(
            query, key, x, key_padding_mask=None, attn_mask=None
        )
        self_attn_output = self.dropout(self_attn_output)
        x = self.layer_norm1(x + self_attn_output)

        # cross attention
        query = self.with_pos_embed(x, query_pos)
        key = encoded_withPos

        num_obj, bs, _ = query.shape
        src_mask = torch.ones(bs, num_obj).bool().to(x.device)
        attn_mask = memory_padding_mask.unsqueeze(1) * src_mask.unsqueeze(2)
        tgt_attn_mask = torch.repeat_interleave(attn_mask, self.num_heads, dim=0)
        cross_attn_output, cross_attn_scores = self.enc_dec_attention(
            query,
            key,
            encoder_output,
            key_padding_mask=memory_padding_mask,
            attn_mask=tgt_attn_mask,
        )
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.layer_norm2(x + cross_attn_output)

        # feed-forward
        ff_output = self.feedforward(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm3(x + ff_output)

        return x


class Decoder(nn.Module):
    def __init__(
        self, num_layers, embed_dim, num_heads, feedforward_dim, dropout_prob=0.1
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, num_heads, feedforward_dim, dropout_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x,
        encoded_withPos,
        encoder_output,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # x: (batch_size, seq_length, embed_dim)
        for layer in self.layers:
            x = layer(
                x,
                encoded_withPos,
                encoder_output,
                memory_padding_mask=memory_padding_mask,
                tgt_attn_mask=tgt_attn_mask,
                query_pos=query_pos,
            )
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout1d(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout1d(dropout)
        self.dropout2 = nn.Dropout1d(dropout)
        self.dropout3 = nn.Dropout1d(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = self.norm1(tgt.permute(1, 0, 2)).permute(1, 0, 2)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = self.norm2(tgt.permute(1, 0, 2)).permute(1, 0, 2)
        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(tgt)).permute(1, 2, 0)).permute(
                2, 0, 1
            )
        )
        tgt = tgt + self.dropout3(tgt2.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = self.norm3(tgt.permute(1, 0, 2)).permute(1, 0, 2)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

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


class FusionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(FusionDecoder, self).__init__()
        decoder_layer = TransformerDecoderLayer(
            input_dim, num_heads, hidden_dim, dropout=0.1, activation="relu"
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)

    def forward(self, object_queries, encoded_sensor_data):
        # Decode using the transformer decoder
        decoder_output = self.transformer_decoder(object_queries, encoded_sensor_data)
        return decoder_output
