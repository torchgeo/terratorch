import math
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchvision.ops import FeaturePyramidNetwork

from terratorch.registry import NECK_REGISTRY, TERRATORCH_NECK_REGISTRY


class Neck(ABC, nn.Module):
    """Base class for Neck

    A neck must must implement `self.process_channel_list` which returns the new channel list.
    """

    def __init__(self, channel_list: list[int]) -> None:
        super().__init__()
        self.channel_list = channel_list

    @abstractmethod
    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return channel_list

    @abstractmethod
    def forward(self, channel_list: list[torch.Tensor], **kwargs) -> list[torch.Tensor]: ...


class NeckSequential(nn.Sequential):
    def forward(self, x, **kwargs):
        for layer in self:
            try:
                x = layer(x, **kwargs)
            except TypeError:
                x = layer(x)
        return x


@TERRATORCH_NECK_REGISTRY.register
class SelectIndices(Neck):
    def __init__(self, channel_list: list[int], indices: list[int]):
        """Select indices from the embedding list

        Args:
            indices (list[int]): list of indices to select.
        """
        super().__init__(channel_list)
        self.indices = indices

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        features = [features[i] for i in self.indices]
        return features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        channel_list = [channel_list[i] for i in self.indices]
        return channel_list


@TERRATORCH_NECK_REGISTRY.register
class AggregateTokens(Neck):
    def __init__(
        self,
        channel_list: list[int],
        pooling: str | int = "mean",
        indices: int | list[int] | None = None,
        index: int | None = -1,  # deprecated,
        drop_cls: bool = False,
        temporal_inputs: bool = False,
    ):
        """Aggregate tokens/patch embeddings to a single embedding per layer. Mainly used for classification models.

        Args:
            pooling (str, int): Pooling method. Options: 'mean', 'max', 'min', 'CLS', or token index (int).
            indices (int | list[int] | None): Layer indices to select from backbone outputs.
            index (int): Deprecated. Select the layer index if multiple outputs are provided. Defaults to -1.
            drop_cls (bool): Whether to drop first token for pooling methods ("mean", "min", "max").
                Intended for ViT-style backbones with a CLS token. Defaults to False.
            temporal_inputs (bool, optional): Used for embedding generation workflows, flag to handle temporal data correctly.
        """
        super().__init__(channel_list)

        self.indices = indices or index  # If indices is not set, use deprecated index, which defaults to -1.
        if isinstance(self.indices, int):  # Wrap int index/ indices to be list.
            self.indices = [self.indices]

        self.pooling = pooling.lower()
        self.latent_dim = [channel_list[i] for i in self.indices]
        self.drop_cls = drop_cls
        self.temporal_inputs = temporal_inputs

        if self.drop_cls and self.pooling == "cls":
            raise ValueError("drop_cls=True is incompatible with pooling='cls'.")

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        aggregated_features = []

        for i, index in enumerate(self.indices):
            feat = features[index] if len(features) > 1 else features[0]

            if self.temporal_inputs:
                # used for inputs with a separate time dim
                timesteps = feat.shape[1]  # shape is (B, T, ...), flatten time dim into B
                feat = rearrange(feat, "b t ... -> (b t) ...")

            if feat.dim() == 4:
                # Assuming spatial grid, flattening spatial dimension
                B, C, H, W = feat.shape
                feat = rearrange(feat, "b c h w -> b (h w) c")

            elif (
                feat.dim() == 5
            ):  # This is fallback for self.temporal_inputs not set, but we know that it is temporal shape
                # Assuming spatiotemporal grid, flattening spatial dimension
                B, C, T, H, W = feat.shape
                feat = rearrange(feat, "b c t h w -> b (h w) t c")

            if self.drop_cls:
                feat = feat[..., 1:, :]

            if isinstance(self.pooling, int):
                # Select token index
                feat_aggregated = feat[..., self.pooling, :]
            elif self.pooling == "cls":
                # Assuming CLS token is on first position
                feat_aggregated = feat[..., 0, :]
            elif self.pooling == "mean":
                feat_aggregated = feat.mean(dim=1)
            elif self.pooling == "max":
                feat_aggregated = feat.max(dim=1).values
            elif self.pooling == "min":
                feat_aggregated = feat.min(dim=1).values
            else:
                raise ValueError(f"Pooling method {self.pooling} not recognized.")

            if self.temporal_inputs:
                # unflatten back to (B, T, ..)
                feat_aggregated = rearrange(feat_aggregated, "(b t) ... -> b t ...", t=timesteps)
            aggregated_features.append(feat_aggregated)

        return aggregated_features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return channel_list


@TERRATORCH_NECK_REGISTRY.register
class PermuteDims(Neck):
    def __init__(self, channel_list: list[int], new_order: list[int]):
        """Permute dimensions of each element in the embedding list

        Args:
            new_order (list[int]): list of indices to be passed to tensor.permute()
        """
        super().__init__(channel_list)
        self.new_order = new_order

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        features = [feat.permute(*self.new_order).contiguous() for feat in features]
        return features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return super().process_channel_list(channel_list)


@TERRATORCH_NECK_REGISTRY.register
class InterpolateToPyramidal(Neck):
    def __init__(self, channel_list: list[int], scale_factor: int = 2, mode: str = "nearest"):
        """Spatially interpolate embeddings so that embedding[i - 1] is scale_factor times larger than embedding[i]

        Useful to make non-pyramidal backbones compatible with hierarachical ones
        Args:
            scale_factor (int): Amount to scale embeddings by each layer. Defaults to 2.
            mode (str): Interpolation mode to be passed to torch.nn.functional.interpolate. Defaults to 'nearest'.
        """
        super().__init__(channel_list)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        out = []
        scale_exponents = list(range(len(features), 0, -1))
        for x, exponent in zip(features, scale_exponents, strict=True):
            out.append(F.interpolate(x, scale_factor=self.scale_factor**exponent, mode=self.mode))

        return out

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return super().process_channel_list(channel_list)


@TERRATORCH_NECK_REGISTRY.register
class MaxpoolToPyramidal(Neck):
    def __init__(self, channel_list: list[int], kernel_size: int = 2):
        """Spatially downsample embeddings so that embedding[i - 1] is scale_factor times smaller than embedding[i]

        Useful to make non-pyramidal backbones compatible with hierarachical ones
        Args:
            kernel_size (int). Base kernel size to use for maxpool. Defaults to 2.
        """
        super().__init__(channel_list)
        self.kernel_size = kernel_size

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        out = []
        scale_exponents = list(range(len(features)))
        for x, exponent in zip(features, scale_exponents, strict=True):
            if exponent == 0:
                out.append(x.clone())
            else:
                out.append(F.max_pool2d(x, kernel_size=self.kernel_size**exponent))

        return out

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return super().process_channel_list(channel_list)


@TERRATORCH_NECK_REGISTRY.register
class ReshapeTokensToImage(Neck):
    def __init__(
        self,
        channel_list: list[int],
        remove_cls_token=True,
        effective_time_dim: int = 1,
        h: int | None = None,
        temporal_inputs: bool = False,
    ):
        """Reshape output of transformer encoder so it can be passed to a conv net.

        Args:
            remove_cls_token (bool, optional): Whether to remove the cls token from the first position.
                Defaults to True.
            effective_time_dim (int, optional): The effective temporal dimension the transformer processes.
                For a ViT, his will be given by `num_frames // tubelet size`. This is used to determine
                the temporal dimension of the embedding, which is concatenated with the embedding dimension.
                For example:
                - A model which processes 1 frame with a tubelet size of 1 has an effective_time_dim of 1.
                    The embedding produced by this model has embedding size embed_dim * 1.
                - A model which processes 3 frames with a tubelet size of 1 has an effective_time_dim of 3.
                    The embedding produced by this model has embedding size embed_dim * 3.
                - A model which processes 12 frames with a tubelet size of 4 has an effective_time_dim of 3.
                    The embedding produced by this model has an embedding size embed_dim * 3.
                Defaults to 1.
            h (int | None):
                You can choose a value for the height of the reshaped image.
                The embedding size will be implicitly discovered from it.
            temporal_inputs (bool, optional): Used for embedding generation workflows, flag to handle temporal data correctly.
        """
        super().__init__(channel_list)
        self.remove_cls_token = remove_cls_token
        self.effective_time_dim = effective_time_dim
        self.h = h
        self.temporal_inputs = temporal_inputs

    def forward(self, features: list[torch.Tensor], image_size=None, **kwargs) -> list[torch.Tensor]:
        out = []
        for x in features:
            if self.temporal_inputs:
                # used for inputs with a separate time dim, effective_time_dim is used for inputs concatenated along channel dim
                timesteps = x.shape[1]  # shape is (B, T, ...), flatten time dim into B
                x = rearrange(x, "b t ... -> (b t) ...")
            if x.dim() >= 4:
                out.append(x)
                continue
            elif x.dim() != 3:
                raise ValueError(f"Expected token tensor (B,N,C) or image tensor (B,C,H,W). Got shape={tuple(x.shape)}")
            else:
                x_no_token = x[:, 1:, :] if self.remove_cls_token else x
                x_no_token = x_no_token.reshape(x.shape[0], -1, x.shape[-1])
                number_of_tokens = x_no_token.shape[1]
                tokens_per_timestep = number_of_tokens // self.effective_time_dim

                # Assume square images first
                h = self.h or math.sqrt(tokens_per_timestep)
                if h - int(h) == 0:
                    h = int(h)
                else:
                    assert image_size is not None, "image_size is not provided for neck ReshapeTokensToImage."
                    # Handle non-square images
                    patch_size = (np.prod(image_size) / tokens_per_timestep) ** 0.5
                    if patch_size % 1:
                        if self.remove_cls_token:
                            warnings.warn(
                                f"Cannot infer grid shape from input tokens ({x.shape[1]}), assuming a cls_token "
                                f"(default setting). Retry ReshapeTokensToImage with remove_cls_token to False. "
                                "Silence this warning with remove_cls_token=False for neck ReshapeTokensToImage."
                            )
                            self.remove_cls_token = False
                            return self.forward(features, image_size, **kwargs)
                        else:
                            raise ValueError(
                                f"Cannot infer grid shape from from input tokens ({x.shape[1]}) with "
                                f"image_size = {image_size} in neck ReshapeTokensToImage. "
                            )
                    img_h, _ = image_size
                    h = int(img_h // patch_size)

                encoded = rearrange(
                    x_no_token,
                    "batch (t h w) e -> batch (t e) h w",
                    batch=x_no_token.shape[0],
                    t=self.effective_time_dim,
                    h=h,
                )

                if self.temporal_inputs:
                    # unflatten back to (B, T, C, H, W)
                    encoded = rearrange(encoded, "(b t) c h w -> b t c h w", t=timesteps)

                out.append(encoded)
        return out

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return super().process_channel_list(channel_list)


@TERRATORCH_NECK_REGISTRY.register
class AddBottleneckLayer(Neck):
    """Add a layer that reduces the channel dimension of the final embedding by half, and concatenates it

    Useful for compatibility with some smp decoders.
    """

    def __init__(self, channel_list: list[int]):
        super().__init__(channel_list)
        self.bottleneck = nn.Conv2d(channel_list[-1], channel_list[-1] // 2, kernel_size=1)

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        new_embedding = self.bottleneck(features[-1])
        features.append(new_embedding)
        return features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return [*channel_list, channel_list[-1] // 2]


@TERRATORCH_NECK_REGISTRY.register
class LearnedInterpolateToPyramidal(Neck):
    """Use learned convolutions to transform the output of a non-pyramidal encoder into pyramidal ones

    Always requires exactly 4 embeddings
    """

    def __init__(self, channel_list: list[int]):
        super().__init__(channel_list)
        if len(channel_list) != 4:
            msg = "This class can only handle exactly 4 input embeddings"
            raise Exception(msg)
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[0], channel_list[0] // 2, 2, 2),
            nn.BatchNorm2d(channel_list[0] // 2),
            nn.GELU(),
            nn.ConvTranspose2d(channel_list[0] // 2, channel_list[0] // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(channel_list[1], channel_list[1] // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.embedding_dim = [channel_list[0] // 4, channel_list[1] // 2, channel_list[2], channel_list[3]]

        if torch.mps.is_available():
            warnings.warn(
                "MPS backend: enforcing .contiguous() to avoid non‑contiguous tensor issues "
                "on M‑series chips; may cause extra copies and slower execution."
            )

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        if torch.mps.is_available():
            # Fix issue on MacBooks
            features = [f.contiguous() for f in features]

        scaled_inputs = []
        scaled_inputs.append(self.fpn1(features[0]))
        scaled_inputs.append(self.fpn2(features[1]))
        scaled_inputs.append(self.fpn3(features[2]))
        scaled_inputs.append(self.fpn4(features[3]))
        return scaled_inputs

    def process_channel_list(self, channel_list: list[int] = None) -> list[int]:
        return [channel_list[0] // 4, channel_list[1] // 2, channel_list[2], channel_list[3]]


@TERRATORCH_NECK_REGISTRY.register
class TemporalPooling(Neck):
    """Pool features over time steps (or any other inputs) concatenated along ``dim``.

    As a neck, each feature has the time steps concatenated along ``dim``, e.g. (B, T*C, H, W) from
    ``ReshapeTokensToImage(effective_time_dim=T)`` or a ``TemporalWrapper`` with ``pooling: concat``.
    It is split into (B, T, C, H, W) and pooled over T. Another ``dim`` pools e.g. token sequences
    (B, N, T*D) with ``dim: 2``. Nothing here is specific to time: modalities concatenated along
    channels (e.g. TerraMind ``merge_method: concat``) are pooled the same way, with
    ``n_timestamps`` set to the number of modalities.

    The ``TemporalWrapper`` pools with this class too, on its stacked (B, T, C, ...) features.

    ```yaml
    necks:
      - name: ReshapeTokensToImage
        effective_time_dim: 4
      - name: TemporalPooling
        n_timestamps: 4
        pooling: linear
    ```

    Args:
        channel_list: Channels per feature, each ``n_timestamps`` times the channels per time step.
        n_timestamps: Number of concatenated time steps (or inputs).
        pooling: How the time steps are merged, per feature:
            "mean" / "max": average / maximum over time.
            "diff": first minus second time step; needs exactly two (after ``subset_lengths``).
            "linear": learned weighted sum over time, ``nn.Linear(T, 1)`` shared across channels
            (T + 1 parameters), like ``multi_temporal_strategy: linear`` in PANGAEA-bench.
            "conv": learned merge of all time steps and channels, a 1x1 ``nn.Conv2d(T*C, C)``
            (T*C*C + C parameters). Needs ``dim=1``.
            "concat" / "keep": no pooling, the features are returned unchanged.
        dim: Dimension along which the time steps are concatenated.
        subset_lengths: If set, average consecutive groups of time steps first (e.g. [2, 3]: the first
            2 and the last 3), then pool over the group means.
    """

    POOLINGS = ("mean", "max", "diff", "linear", "conv", "concat", "keep")

    def __init__(
        self,
        channel_list: list[int],
        n_timestamps: int | None = None,
        pooling: str = "mean",
        dim: int = 1,
        subset_lengths: list[int] | None = None,
    ) -> None:
        super().__init__(channel_list)
        if pooling not in self.POOLINGS:
            msg = f"Unsupported pooling '{pooling}', choose from {list(self.POOLINGS)}."
            raise ValueError(msg)
        if subset_lengths is not None:
            if pooling == "diff" and len(subset_lengths) != 2:
                msg = (
                    f"`diff` pooling requires exactly two subsets in `subset_lengths`, "
                    f"but got {len(subset_lengths)}: {subset_lengths}"
                )
                raise ValueError(msg)
            if n_timestamps is not None and sum(subset_lengths) != n_timestamps:
                msg = (
                    f"The sum of `subset_lengths` must equal `n_timestamps` "
                    f"(got sum={sum(subset_lengths)}, n_timestamps={n_timestamps})."
                )
                raise ValueError(msg)
        self.n_timestamps = n_timestamps
        self.pooling = pooling
        self.dim = dim
        self.subset_lengths = subset_lengths

        if (
            n_timestamps is not None
            and pooling not in ("concat", "keep")
            and any(c % n_timestamps for c in channel_list)
        ):
            msg = f"channel_list={channel_list} is not divisible by n_timestamps={n_timestamps}"
            raise ValueError(msg)
        if pooling in ("linear", "conv"):
            if n_timestamps is None:
                msg = f"'{pooling}' pooling needs n_timestamps"
                raise ValueError(msg)
            n = len(subset_lengths) if subset_lengths else n_timestamps
            if pooling == "linear":
                self.tmap = nn.ModuleList([nn.Linear(n, 1) for _ in channel_list])
            else:
                if dim != 1:
                    msg = f"'conv' pooling needs dim=1, got dim={dim}"
                    raise ValueError(msg)
                self.tmap = nn.ModuleList(
                    [nn.Conv2d(n * (c // n_timestamps), c // n_timestamps, 1) for c in channel_list]
                )

    def pool(self, x: torch.Tensor, level: int = 0, dim: int = 1) -> torch.Tensor:
        """Pool stacked features with the time steps at ``dim`` and the channels at ``dim + 1``."""
        if self.subset_lengths is not None:
            if sum(self.subset_lengths) != x.shape[dim]:
                msg = f"Sum of `subset_lengths` ({sum(self.subset_lengths)}) must equal timesteps ({x.shape[dim]})."
                raise ValueError(msg)
            x = torch.stack([subset.mean(dim=dim) for subset in x.split(self.subset_lengths, dim=dim)], dim=dim)
        if self.pooling == "keep":
            return x
        if self.pooling in ("concat", "conv"):
            x = x.flatten(dim, dim + 1)
            return self.tmap[level](x) if self.pooling == "conv" else x
        if self.pooling == "mean":
            return x.mean(dim=dim)
        if self.pooling == "max":
            return x.max(dim=dim).values
        if self.pooling == "diff":
            if x.shape[dim] != 2:
                msg = (
                    f"`diff` pooling requires exactly two temporal elements "
                    f"(from input or after subset aggregation), got {x.shape[dim]}. "
                    f"Consider to use `subset_lengths` to define two subsets."
                )
                raise ValueError(msg)
            return x.select(dim, 0) - x.select(dim, 1)
        return self.tmap[level](x.movedim(dim, -1)).squeeze(-1)  # linear

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        if self.pooling in ("concat", "keep"):
            return features
        out = []
        for i, x in enumerate(features):
            dim = self.dim % x.dim()
            out.append(self.pool(x.unflatten(dim, (self.n_timestamps, -1)), level=i, dim=dim))
        return out

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        if self.pooling in ("concat", "keep"):
            return channel_list
        return [c // self.n_timestamps for c in channel_list]


@TERRATORCH_NECK_REGISTRY.register
class FeaturePyramidNetworkNeck(Neck):
    """Uses feature pyramid network from torchvision"""

    def __init__(self, channel_list: list[int], out_channel: int = 256, output_ordered_dict: bool = True):
        super().__init__(channel_list)
        self.out_channel = out_channel
        self.fpn = FeaturePyramidNetwork(self.channel_list, self.out_channel)
        self.output_ordered_dict = output_ordered_dict

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if type(features) != "OrderedDict":
            keys = [f"feat{i!s}" for i, x in enumerate(features)]
            features = OrderedDict(zip(keys, features, strict=False))

        reconstructed_features = self.fpn(features)

        if self.output_ordered_dict == False:
            reconstructed_features = list(reconstructed_features.values())

        return reconstructed_features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        channel_list = len(channel_list) * [self.out_channel]
        return channel_list


def build_neck_list(ops: list[dict], channel_list: list[int]) -> tuple[list[Neck], list[int]]:
    neck_list = []
    cur_channel_list = channel_list.copy()
    for cur_op in ops:
        op: Neck = NECK_REGISTRY.build(
            cur_op["name"], cur_channel_list, **{k: v for k, v in cur_op.items() if k != "name"}
        )
        cur_channel_list = op.process_channel_list(cur_channel_list)
        op.channel_list = cur_channel_list
        neck_list.append(op)

    return neck_list, cur_channel_list
