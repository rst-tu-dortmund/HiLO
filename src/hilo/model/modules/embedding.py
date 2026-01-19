import torch
from torch import nn
from torch.nn import Module

import logging
import einops


class SelectiveSinusoidalEmbedding(Module):
    def __init__(self, enabled, pos_channels, N_freq, dim, logscale=True, **kwargs):
        super(SelectiveSinusoidalEmbedding, self).__init__()
        self.enabled = enabled
        self.register_buffer(
            "pos_channels", torch.tensor(pos_channels, dtype=torch.long)
        )
        self.num_pos_channels = len(pos_channels)
        self.N_freq = N_freq
        self.logscale = logscale
        self.dim = dim

        self._input_format = None
        self._invariant_dim_names = None
        self._process_format = None
        self._input_reshape_format = None

        self.additional_channels = 2 * self.num_pos_channels * self.N_freq
        self.insert_index = -1  # -1 means append at the end

        self.funcs = [torch.sin, torch.cos]

        if logscale:
            freq_bands = 2 ** torch.linspace(0, N_freq - 1, N_freq)
        else:
            freq_bands = torch.linspace(1, 2 ** (N_freq - 1), N_freq)

        self.register_buffer("freq_bands", freq_bands)

    def index_shift(self, insert_index, additional_channels):
        # Adjusts used indices based on the insertion index and the number of additional channels
        if (
            insert_index == -1
        ):  # additional channels are appended at the end, thus own indices do not change
            return

        # check all the pos channels indices
        for k, pos_channel in enumerate(self.pos_channels):
            if insert_index < pos_channel:
                self.pos_channels[k] += additional_channels
            elif insert_index == pos_channel:
                logging.warning(
                    f"The positional channel {pos_channel} is already modified by another embedding step."
                )

    def forward(self, x):
        # lazy generate of input and process formats
        if self._input_format is None or self._process_format is None:
            # generate input format to dynamically reshape the input tensor for the process
            # channel dim is always the last one in the process, input may vary
            input_format = [f"d{i}" for i in range(x.ndim)]
            input_format[self.dim] = "C"
            input_format = " ".join(input_format)
            self._input_format = input_format

            # process format therefore is everything stacked and then C
            # implementation currently requires this
            process_format = [f"d{i}" for i in range(x.ndim)]
            process_format.pop(self.dim)
            self._invariant_dim_names = (
                process_format.copy()
            )  # save the invariant dimension names for output reshape, channel dim changes due to frequency bands
            process_format = "(" + " ".join(process_format) + ")" + " C"
            self._process_format = process_format

            self._input_reshape_format = (
                self._input_format + " -> " + self._process_format
            )
            self._output_format = self._process_format + " -> " + self._input_format

        # parse input shapes to extract invariant dimension sizes for output reshape
        input_shapes = einops.parse_shape(x, self._input_format)

        # extract invariant dimension sizes for output reshape
        invariant_shapes = {
            dim_name: input_shapes[dim_name] for dim_name in self._invariant_dim_names
        }

        # reshape as process needs B x C
        x = einops.rearrange(
            x,
            self._input_reshape_format,
        )

        # output start with identity, so we can concatenate the results and keep all information
        out = [x]

        # extract positional channels that are embedded
        pos_channels = x[:, self.pos_channels]

        # we need 3D tensor for the frequency bands multiplication
        if len(pos_channels.shape) == 1:
            pos_channels = pos_channels[:, None, None]
        elif len(pos_channels.shape) == 2:
            pos_channels = pos_channels[..., None]
        else:
            raise ValueError(f"pos_channels shape mismatch, got {pos_channels.shape}")

        # multiply positional channels with frequency bands
        prod = einops.rearrange(
            self.freq_bands[None, None, ...] * pos_channels,
            "... C F -> ... (C F)",
        )

        # apply sinusoidal functions to the product (different frequency bands)
        for func in self.funcs:
            out.append(func(prod))

        # concatenate all results along the last dimension
        out = torch.cat(out, -1)

        # reshape the output to match the input format with extended channel dim
        out_reshaped = einops.rearrange(out, self._output_format, **invariant_shapes)

        return out_reshaped


class OneHotClassEncoding(Module):
    def __init__(self, enabled, class_index, num_classes, dim, **kwargs):
        super(OneHotClassEncoding, self).__init__()
        self.enabled = enabled
        self.register_buffer("class_index", torch.tensor(class_index, dtype=torch.long))

        self.num_classes = num_classes
        self.dim = dim

        self.additional_channels = num_classes  # -1 because we replace class id with one-hot encoding, but +1 because we have padding class
        self.insert_index = class_index

    def index_shift(self, insert_index, additional_channels):
        # Adjusts used indices based on the insertion index and the number of additional channels
        if (
            insert_index == -1
        ):  # additional channels are appended at the end, thus own indices do not change
            return

        if insert_index < self.class_index:
            self.class_index += additional_channels
            self.insert_index += additional_channels
        elif insert_index == self.class_index:
            logging.warning(
                f"The class value is already modified by another embedding step."
            )

    def forward(self, x):
        classes = (
            torch.index_select(x, self.dim, self.class_index[None])
            .to(torch.long)
            .squeeze(-1)
        )
        one_hot_classes = torch.nn.functional.one_hot(
            classes,
            num_classes=self.num_classes
            + 1,  # +1 as padded objects have class id of num_classes
        ).to(torch.float32)

        # replace class id with one-hot encoding
        # 1. The one_hot function adds the new axis at the end. If self.dim is not
        # the last dimension, move the one-hot axis to the correct feature dimension.
        if self.dim % x.dim() != x.dim() - 1:
            one_hot_classes = torch.movedim(one_hot_classes, -1, self.dim)

        # 2. Split the tensor into features before and after the class_id index
        features_before = x.narrow(self.dim, 0, self.class_index)
        features_after = x.narrow(
            self.dim, self.class_index + 1, x.shape[self.dim] - self.class_index - 1
        )

        # 3. Concatenate the parts with the one-hot vector in the middle
        return torch.cat(
            [features_before, one_hot_classes, features_after], dim=self.dim
        )


class DirectionVectorHeadingEncoding(Module):
    def __init__(self, enabled, heading_index, dim, replace_heading=False, **kwargs):
        super(DirectionVectorHeadingEncoding, self).__init__()
        self.enabled = enabled
        self.register_buffer(
            "heading_index", torch.tensor(heading_index, dtype=torch.long)
        )
        self.dim = dim
        self.replace_heading = replace_heading

        self.additional_channels = 1 if replace_heading else 2
        self.insert_index = heading_index if replace_heading else heading_index + 1

    def index_shift(self, insert_index, additional_channels):
        # Adjusts used indices based on the insertion index and the number of additional channels
        if (
            insert_index == -1
        ):  # additional channels are appended at the end, thus own indices do not change
            return

        if insert_index < self.heading_index:
            self.heading_index += additional_channels
            self.insert_index += additional_channels
        elif insert_index == self.heading_index:
            logging.warning(
                f"The heading value is already modified by another embedding step."
            )

    def forward(self, x):
        yaws = (
            torch.index_select(x, self.dim, self.heading_index)
            .to(torch.float32)
            .squeeze(-1)
        )

        heading_dir = torch.stack([torch.cos(yaws), torch.sin(yaws)], dim=-1)

        # replace heading with direction vector
        if self.dim % x.dim() != x.dim() - 1:
            heading_dir = torch.movedim(heading_dir, -1, self.dim)
        # Split the tensor into features before and after the heading index
        features_before = x.narrow(
            self.dim,
            0,
            self.heading_index if self.replace_heading else self.heading_index + 1,
        )
        features_after = x.narrow(
            self.dim, self.heading_index + 1, x.shape[self.dim] - self.heading_index - 1
        )

        # Concatenate the parts with the heading direction vector in the middle
        return torch.cat([features_before, heading_dir, features_after], dim=self.dim)


class Embedding(Module):
    def __init__(self, cfg):
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.enabled = cfg.get("enabled", True)

        if not self.enabled:
            self.additional_channels = 0
            return

        embeddings = cfg.get("embeddings", [])

        self.emb_steps = nn.ModuleList()
        additional_channels = torch.zeros(len(embeddings), dtype=torch.long)

        for k, emb_step in enumerate(embeddings):
            emb_type, emb_kwargs = emb_step["type"].lower(), emb_step["kwargs"]

            if "positional_embedding" == emb_type:
                emb = SelectiveSinusoidalEmbedding(**emb_kwargs)
                self.emb_steps.append(emb)

            elif "one_hot_class_encoding" == emb_type:
                emb = OneHotClassEncoding(**emb_kwargs)
                self.emb_steps.append(emb)

            elif "direction_vector_heading_encoding" == emb_type:
                emb = DirectionVectorHeadingEncoding(**emb_kwargs)
                self.emb_steps.append(emb)

            else:
                raise ValueError(
                    f"Unknown embedding type: {emb_type}. Supported types are: positional_embedding, one_hot_class_encoding, direction_vector_heading_encoding."
                )

            additional_channels[k] = emb.additional_channels

        self.additional_channels = additional_channels.sum().item()

        # for k, curr_emb in enumerate(self.emb_steps):
        #     for next_emb in self.emb_steps[k + 1:]:
        #         next_emb.index_shift(curr_emb.insert_index, curr_emb.additional_channels)

    def forward(self, x):
        if not self.enabled:
            return x

        for emb_step in self.emb_steps:
            if emb_step.enabled:
                x = emb_step(x)

        return x
