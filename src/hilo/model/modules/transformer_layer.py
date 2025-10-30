import einops
import numpy as np
import torch

from torch import Tensor
from torch.nn import Module, MultiheadAttention, Linear, LayerNorm, Dropout
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn

from typing import Optional, Callable, Any, Union

from collections import OrderedDict


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        use_he_init=False,
        learned_inv_temperature: bool = False,
        learned_inv_temperature_init: float = 1.0,
        multi_query_attention: int = 1,  # <= 1 is off, >1 is number of multi-queries
        multi_query_reduction: str = "max",
        pairwise_aggregation: bool = False,
        init_in_proj_weight_identity: bool = False,
        **kwargs: Any,  # for backward compatibility
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.batch_first = batch_first
        multi_query_attention = max(multi_query_attention, 1)
        self.multi_query_attention = multi_query_attention
        self.multi_query_reduction = multi_query_reduction.lower()
        self.pairwise_aggregation = pairwise_aggregation

        if pairwise_aggregation:
            self.pair_aggr = torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            "lin1",
                            Linear(
                                2 * d_model,
                                d_model,
                                bias=bias,
                                **factory_kwargs,
                            ),
                        ),
                        (
                            "norm",
                            LayerNorm(
                                d_model,
                                eps=layer_norm_eps,
                                bias=bias,
                                **factory_kwargs,
                            ),
                        ),
                        ("activation", torch.nn.ReLU()),
                        ("dropout", Dropout(dropout)),
                        (
                            "lin2",
                            Linear(
                                d_model,
                                d_model,
                                bias=bias,
                                **factory_kwargs,
                            ),
                        ),
                    ]
                )
            )

        if self.multi_query_attention > 1:
            if batch_first:
                einops_split_format = "b s (m q) -> b s m q"
                einops_cat_format = "b s m q -> b (s m) q"
                einops_rearrange_format = "b (s m) q -> b s q m"
            else:
                einops_split_format = "s b (m q) -> s b m q"
                einops_cat_format = "s b m q -> (s m) b q"
                einops_rearrange_format = "(s m) b q -> s b q m"

            self.query_to_multi_query = torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            "lin1",
                            Linear(
                                d_model,
                                d_model * self.multi_query_attention,
                                bias=bias,
                                **factory_kwargs,
                            ),
                        ),
                        (
                            "norm",
                            LayerNorm(
                                d_model * self.multi_query_attention,
                                eps=layer_norm_eps,
                                bias=bias,
                                **factory_kwargs,
                            ),
                        ),
                        ("activation", torch.nn.ReLU()),
                        ("dropout", Dropout(dropout)),
                        (
                            "lin2",
                            Linear(
                                d_model * self.multi_query_attention,
                                d_model * self.multi_query_attention,
                                bias=bias,
                                **factory_kwargs,
                            ),
                        ),
                        (
                            "split",
                            einops.layers.torch.Rearrange(
                                einops_split_format,
                                m=self.multi_query_attention,
                                q=d_model,
                            ),
                        ),
                    ]
                )
            )
            # torch.nn.init.zeros_(self.query_to_multi_query.lin2.weight)
            # if self.query_to_multi_query.lin2.bias is not None:
            #     torch.nn.init.zeros_(self.query_to_multi_query.lin2.bias)

            self.cat_multi_queries = einops.layers.torch.Rearrange(einops_cat_format)
            self.move_multi_queries_last_dim = einops.layers.torch.Rearrange(
                einops_rearrange_format, m=self.multi_query_attention + 1
            )

        self.learned_temperature = learned_inv_temperature
        self.log_inv_temperature = None
        if learned_inv_temperature:
            self.log_inv_temperature = torch.nn.Parameter(
                torch.tensor(
                    np.log(learned_inv_temperature_init), dtype=torch.float32
                ).repeat(multi_query_attention),
                requires_grad=True,
            )

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        if init_in_proj_weight_identity:
            self.self_attn.in_proj_weight = torch.nn.Parameter(
                torch.eye(
                    d_model, dtype=dtype, device=device, requires_grad=True
                ).repeat(3, 1)
            )

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)

        self.dropout = Dropout(dropout)

        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        if use_he_init:
            torch.nn.init.kaiming_uniform_(
                self.linear1.weight, mode="fan_in", nonlinearity="relu"
            )
            torch.nn.init.kaiming_uniform_(
                self.linear2.weight, mode="fan_in", nonlinearity="relu"
            )

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
        src_pos_enc: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        why_not_sparsity_fast_path = ""
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (
            src_key_padding_mask is not None or src_mask is not None
        ):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                (x.device.type in _supported_device_type) for x in tensor_args
            ):
                why_not_sparsity_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"{_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(
                    src_mask, src_key_padding_mask, src
                )
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        return self._forward(
            src, src_mask, src_key_padding_mask, is_causal, need_weights, src_pos_enc
        )

    def _forward(
        self, src, src_mask, src_key_padding_mask, is_causal, need_weights, src_pos_enc
    ):
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x_, a = self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
                need_weights=need_weights,
                src_pos_enc=src_pos_enc,
            )
            x = x + x_
            x = x + self._ff_block(self.norm2(x))
        else:
            x_, a = self._sa_block(
                x,
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
                need_weights=need_weights,
                src_pos_enc=src_pos_enc,
            )
            x = self.norm1(x + x_)
            x = self.norm2(x + self._ff_block(x))

        return x, a

    # self-attention block
    def _sa_block(
        self,
        src: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        need_weights: bool = False,
        src_pos_enc: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.multi_query_attention > 1:
            # each query generates self.multi_query_attention queries
            attn_mask = einops.repeat(
                attn_mask, "s1 s2 -> (s1 m) s2", m=self.multi_query_attention + 1
            )  # (batch, seq_len, multi_query)
            multi_queries = self.query_to_multi_query(x)  # b s m q
            multi_queries = x[..., None, :] + 0.1 * multi_queries  # b s m q
            multi_queries = torch.cat(
                [x[..., None, :], multi_queries], dim=-2
            )  # b s m+1 q

            if self.log_inv_temperature is not None:
                inv_temperature = torch.exp(self.log_inv_temperature)
                scaled_multi_queries = (
                    multi_queries * inv_temperature[None, None, :, None]
                )
                q = self.cat_multi_queries(scaled_multi_queries)
            else:
                q = self.cat_multi_queries(multi_queries)

        else:
            if self.log_inv_temperature is not None:
                inv_temperature = torch.exp(self.log_inv_temperature)
                q = x * inv_temperature
            else:
                q = x

        k = x.clone()
        if src_pos_enc is not None:
            q = q + src_pos_enc
            k = k + src_pos_enc

        x, a = self.self_attn(
            q,  # Query
            k,  # Key
            x,  # Value
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=is_causal,
        )

        if self.multi_query_attention > 1:
            x = self.move_multi_queries_last_dim(x)

            if self.multi_query_reduction == "max":
                x = torch.amax(x, dim=-1)  # reduce to original sequence length
            elif self.multi_query_reduction == "mean":
                x = torch.mean(x, dim=-1)
            elif self.multi_query_reduction == "sum":
                x = torch.sum(x, dim=-1)

            elif self.multi_query_reduction == "debug":
                x = x[..., 0]
            else:
                raise ValueError(f"Unknown reduction type {self.multi_query_reduction}")

            if a is not None:
                a = einops.rearrange(
                    a, "b (s m) n -> b s n m", m=self.multi_query_attention + 1
                )

                if self.multi_query_reduction == "max":
                    a = torch.amax(a, dim=-1)
                elif self.multi_query_reduction == "mean":
                    a = torch.mean(a, dim=-1)
                elif self.multi_query_reduction == "sum":
                    a = torch.norm(torch.sum(a, dim=-1), dim=-1)

                elif self.multi_query_reduction == "debug":
                    a = a[..., : a.shape[1]]
                else:
                    raise ValueError(
                        f"Unknown reduction type {self.multi_query_reduction}"
                    )

        if self.pairwise_aggregation:
            x = self.pair_aggr(torch.cat((x, src), dim=-1))

        return self.dropout1(x), a

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        use_he_init=False,
        **kwargs: Any,  # catch unused kwargs
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        if use_he_init:
            torch.nn.init.kaiming_uniform_(
                self.linear1.weight, mode="fan_in", nonlinearity="relu"
            )
            torch.nn.init.kaiming_uniform_(
                self.linear2.weight, mode="fan_in", nonlinearity="relu"
            )

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        tgt_pos_enc: Optional[Tensor] = None,
        memory_pos_enc: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
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
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                tgt_mask,
                tgt_key_padding_mask,
                tgt_is_causal,
                tgt_pos_enc,
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
                tgt_pos_enc,
                memory_pos_enc,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x, tgt_mask, tgt_key_padding_mask, tgt_is_causal, tgt_pos_enc
                )
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                    tgt_pos_enc,
                    memory_pos_enc,
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        src_pos_enc: Optional[Tensor] = None,
    ) -> Tensor:
        q = x.clone()
        k = x.clone()
        v = x
        if src_pos_enc is not None:
            q = q + src_pos_enc
            k = k + src_pos_enc
        x = self.self_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        src_pos_enc: Optional[Tensor] = None,
        mem_pos_enc: Optional[Tensor] = None,
    ) -> Tensor:
        q = x.clone()
        k = mem.clone()
        v = mem

        if src_pos_enc is not None:
            q = q + src_pos_enc

        if mem_pos_enc is not None:
            k = k + mem_pos_enc

        x = self.multihead_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
