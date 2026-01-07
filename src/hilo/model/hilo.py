import torch
from torch import nn
from torch.nn import Module

import math
import logging
import einops

from hilo.model.base_model import BaseModel
from hilo.model.modules.embedding import Embedding
from hilo.model.modules.mlp import MLP
from hilo.model.modules.transformer import TransformerEncoder, TransformerDecoder

from hilo.utils.data.normalization import Normalization


class HiLO(BaseModel):
    def __init__(self, cfg):
        super(HiLO, self).__init__()
        # Initialize model components based on model cfg
        # This is a placeholder implementation
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        self.no_mma_tf_masks = cfg.get("no_mma_tf_masks", False)
        if self.no_mma_tf_masks:
            self.logger.warning("No multi-modal attention transformer masks will be used.")
            
        self.no_fusion_tf_masks = cfg.get("no_fusion_tf_masks", False)
        if self.no_fusion_tf_masks:
            self.logger.warning("No fusion decoder transformer masks will be used.")

        self.detection_normalization = Normalization(cfg["detection_normalization"])
        self.detection_embedding = Embedding(cfg["detection_embedding"])
        cfg["detection_input_mlp"][
            "in_channels"
        ] += self.detection_embedding.additional_channels
        self.detection_input_mlp = MLP(cfg["detection_input_mlp"])

        self.multi_modal_attention_tf = TransformerEncoder(cfg["multi_modal_attention"])

        self.fusion_decoder_tf = TransformerDecoder(cfg["fusion_decoder"])
        self.number_of_queries = cfg["fusion_decoder"]["num_queries"]
        fusion_queries = torch.randn(
            (cfg["fusion_decoder"]["num_queries"], 1, cfg["fusion_decoder"]["d_model"])
        )

        self.tf_encoder_decoder_interface = None
        if cfg["multi_modal_attention"]["d_model"] != cfg["fusion_decoder"]["d_model"]:
            self.tf_encoder_decoder_interface = torch.nn.Linear(
                cfg["multi_modal_attention"]["d_model"],
                cfg["fusion_decoder"]["d_model"],
            )

        fusion_queries = fusion_queries / fusion_queries.abs().max()
        fusion_queries.requires_grad = True
        self.register_parameter("fusion_queries", torch.nn.Parameter(fusion_queries, requires_grad=True))

        self.box_regression_head = MLP(cfg["box_regression_head"])
        denorm_reg_output_idcs = torch.tensor(
            self.cfg["box_regression_head"]["denormalization"]["output_indices"],
            dtype=torch.long,
        )
        self.register_buffer("denorm_reg_output_idcs", denorm_reg_output_idcs)

        reg_output_norm_idcs = self.cfg["box_regression_head"]["denormalization"][
            "norm_indices"
        ]
        self.register_buffer(
            "reg_output_norm_idcs", torch.tensor(reg_output_norm_idcs, dtype=torch.long)
        )

        self.classification_head = MLP(cfg["classification_head"])

    def shared_object_encoding(self, x, mask):
        # Shared object encoding across sensors and detections
        b, s, n, c = x.shape  # batch, sensors, num_detections, channels

        x_ = einops.rearrange(x, "b s n c -> (b s n) c")
        mask_ = einops.rearrange(mask, "b s n -> (b s n)")

        x_norm = self.detection_normalization(x_, mask_)
        x_emb = self.detection_embedding(x_norm)
        x_enc = self.detection_input_mlp(x_emb)

        x_out = einops.rearrange(x_enc, "(b s n) c -> b s n c", b=b, s=s, n=n)

        return x_out

    def multi_modal_attention(self, x, mask, pe_feat):
        # Multi-modal attention across sensors
        b, s, n, c = x.shape  # batch, sensors, num_detections, channels

        x_ = einops.rearrange(x, "b s n c -> (s n) b c")
        pe_feat_ = einops.rearrange(pe_feat, "b s n c -> b (s n) c")
        mask_ = einops.rearrange(mask, "b s n -> b (s n)")

        if self.cfg["multi_modal_attention"].get("use_positional_encoding", False):
            pe = self._get_positional_embeddings(
                x_,
                pos_enc_feat=pe_feat_,
                tf_dim_pattern="n b c",
                num_heads=(
                    self.cfg["multi_modal_attention"]["nhead"]
                    if self.cfg["multi_modal_attention"].get(
                        "positional_encoding_per_head", False
                    )
                    else None
                ),
                scale_factor=self.cfg["multi_modal_attention"].get(
                    "positional_encoding_scale_factor", 1.0
                ),
            )
            x_ = x_ + pe

        sequence_mask = None
        if self.cfg["multi_modal_attention"].get("exclude_self_attention", False):
            sequence_mask = torch.eye(
                s * n, device=x.device
            ).bool()  # true values will be masked

        x_mma, attn_weights = self.multi_modal_attention_tf(
            x_,
            src_key_padding_mask=~mask_ if not self.no_mma_tf_masks else None,
            is_causal=False,
            mask=sequence_mask if not self.no_mma_tf_masks else None,
        )

        x_mma = einops.rearrange(x_mma, "(s n) b c -> b s n c", b=b, s=s, n=n)

        return x_mma

    def fusion_decoding(self, x, mask, pe_feat):
        # Fusion decoding to get fused object representations
        b, s, n, c = x.shape  # batch, sensors, num_detections, channels

        fusion_queries = einops.repeat(
            self.fusion_queries,
            "q 1 c -> q b c",
            b=b,
        )

        x_ = einops.rearrange(x, "b s n c -> (s n) b c")

        if self.tf_encoder_decoder_interface is not None:
            x_ = self.tf_encoder_decoder_interface(x_)

        pe_feat_ = einops.rearrange(pe_feat, "b s n c -> b (s n) c")
        mask_ = einops.rearrange(mask, "b s n -> b (s n)")

        if self.cfg["fusion_decoder"].get("use_positional_encoding", False):
            pe = self._get_positional_embeddings(
                x_,
                pos_enc_feat=pe_feat_,
                tf_dim_pattern="n b c",
                num_heads=(
                    self.cfg["fusion_decoder"]["nhead"]
                    if self.cfg["fusion_decoder"].get(
                        "positional_encoding_per_head", False
                    )
                    else None
                ),
                scale_factor=self.cfg["fusion_decoder"].get(
                    "positional_encoding_scale_factor", 1.0
                ),
            )
            x_ = x_ + pe

        x_dec = self.fusion_decoder_tf(
            tgt=fusion_queries,
            memory=x_,
            memory_key_padding_mask=~mask_ if not self.no_fusion_tf_masks else None,
        )

        x_dec = einops.rearrange(x_dec, "q b c -> b q c")

        return x_dec

    def forward(self, x, mask):
        """Forward pass for the HiLO model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            _type_: _description_
        """
        b, s, n, c = x.shape  # batch, sensors, num_detections, channels
        pe_feat = x[
            ..., :2
        ]  # assuming first two channels are positional features (x, y)

        # encode detections
        x_enc = self.shared_object_encoding(x, mask)

        # multi-modal attention
        x_mma = self.multi_modal_attention(x_enc, mask, pe_feat)

        # decode fused object representations
        x_dec = self.fusion_decoding(x_mma, mask, pe_feat)

        # predict boxes
        x_dec = einops.rearrange(x_dec, "b q c -> (b q) c")
        box_pred_norm = self.box_regression_head(x_dec)
        box_pred_norm = torch.cat([
            box_pred_norm[..., :2],
            torch.exp(box_pred_norm[..., 2:4]),
            box_pred_norm[..., 4:],
        ], dim=-1)
        box_pred = self.detection_normalization.denormalize(
            box_pred_norm,
            feature_idcs=self.denorm_reg_output_idcs,
            norm_idcs=self.reg_output_norm_idcs,
        )
        box_pred = einops.rearrange(box_pred, "(b q) c -> b q c", b=b)

        # predict classes
        class_pred = self.classification_head(x_dec)
        class_pred = einops.rearrange(class_pred, "(b q) c -> b q c", b=b)

        output = self.prepare_output(box_pred, class_pred)

        return output

    def _get_positional_embeddings(
        self, x, pos_enc_feat, tf_dim_pattern, num_heads=None, scale_factor=1.0
    ):
        with torch.no_grad():
            tf_dims = einops.parse_shape(
                x, tf_dim_pattern
            )  # yields b n c (either b n c or n b c), c is always last!
            c_head = tf_dims["c"] // num_heads if num_heads is not None else tf_dims["c"]
            pos_enc_in = pos_enc_feat.detach()  # b n c
            n_elem = pos_enc_in.shape[-1] * 2
            n_freq = math.ceil(c_head / n_elem)

            i = torch.arange(n_freq, dtype=torch.float32, device=x.device)
            denom = self.cfg["max_range"] ** (2 * i / c_head)

            pos_enc_in.unsqueeze_(-1)  # b n c 1
            # x = pos_enc_in[..., 0, None]
            # y = pos_enc_in[..., 1, None]

            # pe = torch.zeros(
            #     (tf_dims["b"], tf_dims["n"], c_head),
            #     device=x.device,
            #     dtype=x.dtype,
            # )
            pe_s = torch.sin(pos_enc_in / denom)  # b n pe_feat n_freq
            pe_c = torch.cos(pos_enc_in / denom)  # b n pe_feat n_freq

            pe = torch.cat([pe_s, pe_c], dim=-1)  # b n pe_feat 2*n_freq
            pe = einops.rearrange(
                pe,
                "b n c f -> b n (c f)",
            )  # b n (pe_feat*2*n_freq)

            # pe[..., 0::4] = torch.sin(x / denom)
            # pe[..., 1::4] = torch.cos(x / denom)
            # pe[..., 2::4] = torch.sin(y / denom)
            # pe[..., 3::4] = torch.cos(y / denom)

            if num_heads is not None:
                pe = einops.repeat(
                    pe,
                    "b n c -> b n (h c)",
                    h=num_heads,
                )

            pe = pe[..., : tf_dims["c"]]
            pos_enc = einops.rearrange(pe, "b n c -> " + tf_dim_pattern) * scale_factor

        return pos_enc
