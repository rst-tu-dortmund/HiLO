import warnings
import torch
from torch import Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from typing import Optional
from torch.nn.modules.transformer import _get_seq_len, _detect_is_causal_mask

from hilo.model.modules.transformer_layer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)


def _get_clones_custom(module, N, layer_kwargs):
    return ModuleList([module(**layer_kwargs) for i in range(N)])


class TransformerEncoder(Module):
    # taken from pytorch with minor modifications
    r"""TransformerEncoder is a stack of N encoder layers.

    Users can build the BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        cfg: dict,
        # encoder_layer: Module,
        # layer_kwargs: dict,
        # num_layers: int,
        # norm: Optional[Module] = None,
        # enable_nested_tensor: bool = True,
        # mask_check: bool = True,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        encoder_layer = TransformerEncoderLayer
        num_layers = cfg["num_layers"]
        norm = cfg.get("norm", None)

        enable_nested_tensor = cfg.get("enable_nested_tensor", True)
        mask_check = cfg.get("mask_check", True)

        self.layers = _get_clones_custom(encoder_layer, num_layers, cfg)
        encoder_layer = self.layers[0]
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ""
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first:
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn.batch_first was not True"
                + "(use batch_first for better inference performance)"
            )
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
            )
        elif encoder_layer.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = (
                f"{enc_layer}.activation_relu_or_gelu was not True"
            )
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps):
            why_not_sparsity_fast_path = (
                f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
            )
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(
                f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}"
            )
            self.use_nested_tensor = False

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        need_attention_weights: Optional[bool] = False,
        src_pos_enc: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (
            (not hasattr(self, "mask_check")) or self.mask_check
        ) and not torch._nested_tensor_from_mask_left_aligned(
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        weights = []
        for mod in self.layers:
            output, attn = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
                need_weights=need_attention_weights,
                src_pos_enc=src_pos_enc,
            )
            weights.append(attn)

        if convert_to_nested:
            output = output.to_padded_tensor(0.0, src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output, weights


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        cfg: dict,
        # decoder_layer: "TransformerDecoderLayer",
        # layer_kwargs: dict,
        # num_layers: int,
        # norm: Optional[Module] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        decoder_layer = TransformerDecoderLayer

        num_layers = cfg["num_layers"]
        norm = cfg.get("norm", None)

        self.layers = _get_clones_custom(decoder_layer, num_layers, cfg)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
        tgt_pos_enc: Optional[Tensor] = None,
        memory_pos_enc: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
                tgt_pos_enc=tgt_pos_enc,
                memory_pos_enc=memory_pos_enc,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
