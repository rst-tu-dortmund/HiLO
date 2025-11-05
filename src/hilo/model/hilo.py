import torch
from torch.nn import Module

import math

import einops

from hilo.model.modules.embedding import Embedding
from hilo.model.modules.mlp import MLP
from hilo.model.modules.transformer import TransformerEncoder, TransformerDecoder

from hilo.utils.data.normalization import Normalization


class HiLO(Module):
    def __init__(self, cfg):
        super(HiLO, self).__init__()
        # Initialize model components based on model cfg
        # This is a placeholder implementation
        self.cfg = cfg

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
        self.register_buffer("fusion_queries", fusion_queries)

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

        x_ = self.detection_normalization(x_, mask_)
        x_ = self.detection_embedding(x_)
        x_ = self.detection_input_mlp(x_)

        x = einops.rearrange(x_, "(b s n) c -> b s n c", b=b, s=s, n=n)

        return x

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
            src_key_padding_mask=~mask_,
            is_causal=False,
            mask=sequence_mask,
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
            memory_key_padding_mask=~mask_,
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

        # predict boxes and classes
        x_dec = einops.rearrange(x_dec, "b q c -> (b q) c")
        box_pred_norm = self.box_regression_head(x_dec)
        box_pred = self.detection_normalization.denormalize(
            box_pred_norm,
            feature_idcs=self.denorm_reg_output_idcs,
            norm_idcs=self.reg_output_norm_idcs,
        )
        box_pred = einops.rearrange(box_pred, "(b q) c -> b q c", b=b)

        class_pred = self.classification_head(x_dec)
        class_pred = einops.rearrange(class_pred, "(b q) c -> b q c", b=b)

        output = self.prepare_output(box_pred, class_pred)

        return output

    def prepare_output(self, box_pred, class_pred):
        # compute yaw from sin and cos components
        yaw = torch.atan2(
            box_pred[..., -2], box_pred[..., -1]
        )  # assuming last two channels are sin(yaw) and cos(yaw)
        # box_pred = torch.cat([box_pred[..., :-2], yaw.unsqueeze(-1)], dim=-1)
        output = {
            "centers": box_pred[..., :2],
            "extents": box_pred[..., 2:4],
            "yaws": yaw,
            "heading_directions": box_pred[..., -2:],
            "velocities": box_pred[..., 4:6],
            "bboxes": torch.cat([box_pred[..., :6], yaw.unsqueeze(-1)], dim=-1),
            "class_logits": class_pred,
            "class_probs": torch.softmax(class_pred, dim=-1),
            "class_ids": torch.argmax(class_pred, dim=-1).to(torch.int64),
        }

        if not self.training:
            output = self._filter_inference_output(output)

        return output

    def _filter_inference_output(self, output):
        B, N, C = output["class_probs"].shape
        device = output["class_probs"].device
        ##### applying the NMS free filtering approach from "End-to-End Multi-Object Detection with Transformers"
        # getting the class probabilities without the "no object" class and individually (sigmoid)
        indv_cls_probs = torch.sigmoid(output["class_probs"][..., :-1])
        indv_cls_ids = (
            torch.arange(indv_cls_probs.shape[-1], device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(B, N, 1)
        )

        indv_cls_probs_flat = einops.rearrange(
            indv_cls_probs, "b n c -> b (n c)"
        )  # shape: (B, N*(num_classes-1))

        indv_cls_ids_flat = einops.rearrange(
            indv_cls_ids, "b n c -> b (n c)"
        )  # shape: (B, N*(num_classes-1))

        topk_prob, topk_prob_idx = torch.topk(
            indv_cls_probs_flat, k=self.number_of_queries, dim=-1
        )
        topk_obj_idx = topk_prob_idx // (C - 1)

        output_filtered = {
            k: v[torch.arange(B, device=device).unsqueeze(-1), topk_obj_idx]
            for k, v in output.items()
        }

        # apply thresholding
        keep_mask = topk_prob >= self.cfg.get("inference_score_threshold", 0.1)
        output_filtered["mask"] = keep_mask

        return output_filtered

    def _get_positional_embeddings(
        self, x, pos_enc_feat, tf_dim_pattern, num_heads=None, scale_factor=1.0
    ):
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

    def get_parameter_groups(self, weight_decay=0.01, bias_norm_decay=True):
        # returns parameter groups for optimizer with optional weight decay settings
        no_decay = []
        decay = []

        if bias_norm_decay:
            return [
                {
                    "params": self.parameters(),
                    "weight_decay": weight_decay,
                }
            ]
        else:
            for name, param in self.named_parameters():
                if "bias" in name or "norm" in name or "bn" in name or "Norm" in name:
                    no_decay.append(param)
                else:
                    decay.append(param)

        return [
            {
                "params": decay,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ]
