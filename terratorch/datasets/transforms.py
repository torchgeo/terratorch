# Copyright contributors to the Terratorch project

from typing import Dict, Union

import albumentations as A
import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import BasicTransform, Compose, ImageOnlyTransform
from einops import rearrange
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from torch import nn

N_DIMS_FOR_TEMPORAL = 4
N_DIMS_FLATTENED_TEMPORAL = 3


def kornia_augmentations_to_callable_with_dict(
    augmentations: list[GeometricAugmentationBase2D | K.VideoSequential] | None = None,
):
    if augmentations is None:
        return lambda x: x
    # if first augmentiaion is VideoSequential (multi-temporal), add the rest to video sequence
    if isinstance(augmentations[0], K.VideoSequential):
        augmentations = K.AugmentationSequential(
            K.VideoSequential(*augmentations[1:], data_format="BCTHW", same_on_frame=True),
            data_keys=None,
            keepdim=True,
        )
    else:
        augmentations = K.AugmentationSequential(
            *augmentations,
            data_keys=None,
            keepdim=True,
        )

    def fn(data):
        return augmentations(data)

    return fn


def albumentations_to_callable_with_dict(albumentation: list | None = None):
    if albumentation is None:
        return lambda x: x
    albumentation = Compose(albumentation)

    def fn(data):
        return albumentation(**data)

    return fn


def default_non_image_transform(array):
    if array.dtype in (float, int):
        return torch.from_numpy(array)
    else:
        return array


class Padding(ImageOnlyTransform):
    """Padding to adjust (slight) discrepancies between input images"""

    def __init__(self, input_shape: list = None):
        super().__init__(True, 1)
        self.input_shape = input_shape

    def apply(self, img, **params):

        shape = img.shape[-2:]
        pad_values_ = [j - i for i, j in zip(shape, self.input_shape)]

        if all([i % 2 == 0 for i in pad_values_]):
            pad_values = sum([[int(j / 2), int(j / 2)] for j in pad_values_], [])
        else:
            pad_values = sum([[0, j] for j in pad_values_], [])

        return F.pad(img, pad_values)

    def get_transform_init_args_names(self):
        return ()


class FlattenTemporalIntoChannels(ImageOnlyTransform):
    """
    FlattenTemporalIntoChannels is an image transformation that flattens the temporal dimension into the channel dimension.

    This transform rearranges an input tensor with a temporal dimension into one where the time and channel dimensions
    are merged. It expects the input to have a fixed number of dimensions defined by N_DIMS_FOR_TEMPORAL.
    """

    def __init__(self):
        """
        Initialize the FlattenTemporalIntoChannels transform.
        """
        super().__init__(True, 1)

    def apply(self, img, **params):
        if len(img.shape) != N_DIMS_FOR_TEMPORAL:
            msg = f"Expected input temporal sequence to have {N_DIMS_FOR_TEMPORAL} dimensions, but got {len(img.shape)}"
            raise Exception(msg)
        rearranged = rearrange(img, "time height width channels -> height width (time channels)")
        return rearranged

    def get_transform_init_args_names(self):
        return ()


class UnflattenTemporalFromChannels(ImageOnlyTransform):
    """
    UnflattenTemporalFromChannels is an image transformation that restores the temporal dimension from the channel dimension.

    This transform is typically applied after converting images to a channels-first format (e.g., after ToTensorV2)
    and rearranges the flattened temporal information back into separate time and channel dimensions.
    """

    def __init__(self, n_timesteps: int | None = None, n_channels: int | None = None):
        super().__init__(True, 1)
        """
        Initialize the UnflattenTemporalFromChannels transform.

        Args:
            n_timesteps (int | None): The number of time steps. Must be provided if n_channels is not provided.
            n_channels (int | None): The number of channels per time step. Must be provided if n_timesteps is not provided.

        Raises:
            Exception: If neither n_timesteps nor n_channels is provided.
        """
        if n_timesteps is None and n_channels is None:
            msg = "One of n_timesteps or n_channels must be provided"
            raise Exception(msg)
        self.additional_info = {"channels": n_channels} if n_channels else {"time": n_timesteps}

    def apply(self, img, **params):
        if len(img.shape) != N_DIMS_FLATTENED_TEMPORAL:
            msg = f"Expected input temporal sequence to have {N_DIMS_FLATTENED_TEMPORAL} dimensions\
                , but got {len(img.shape)}"
            raise Exception(msg)

        rearranged = rearrange(
            img, "(time channels) height width -> channels time height width", **self.additional_info
        )
        return rearranged

    def get_transform_init_args_names(self):
        return ("n_timesteps", "n_channels")


class FlattenSamplesIntoChannels(ImageOnlyTransform):
    """
    FlattenSamplesIntoChannels is an image transformation that merges the sample (and optionally temporal) dimensions into the channel dimension.

    This transform rearranges an input tensor by flattening the sample dimension, and if specified, also the temporal dimension,
    thereby concatenating these dimensions into a single channel dimension.
    """

    def __init__(self, time_dim: bool = True):
        """
        Initialize the FlattenSamplesIntoChannels transform.

        Args:
            time_dim (bool): If True, the temporal dimension is included in the flattening process. Default is True.
        """
        super().__init__(True, 1)
        self.time_dim = time_dim

    def apply(self, img, **params):
        if self.time_dim:
            rearranged = rearrange(img, "samples time height width channels -> height width (samples time channels)")
        else:
            rearranged = rearrange(img, "samples height width channels -> height width (samples channels)")
        return rearranged

    def get_transform_init_args_names(self):
        return ()


class UnflattenSamplesFromChannels(ImageOnlyTransform):
    """
    UnflattenSamplesFromChannels is an image transformation that restores the sample (and optionally temporal) dimensions from the channel dimension.

    This transform is designed to reverse the flattening performed by FlattenSamplesIntoChannels and is typically applied
    after converting images to a channels-first format.
    """

    def __init__(
        self,
        time_dim: bool = True,
        n_samples: int | None = None,
        n_timesteps: int | None = None,
        n_channels: int | None = None,
    ):
        """
        Initialize the UnflattenSamplesFromChannels transform.

        Args:
            time_dim (bool): If True, the temporal dimension is considered during unflattening.
            n_samples (int | None): The number of samples.
            n_timesteps (int | None): The number of time steps.
            n_channels (int | None): The number of channels per time step.

        Raises:
            Exception: If time_dim is True and fewer than two of n_channels, n_timesteps, and n_samples are provided.
            Exception: If time_dim is False and neither n_channels nor n_samples is provided.
        """
        super().__init__(True, 1)

        self.time_dim = time_dim
        if self.time_dim:
            if bool(n_channels) + bool(n_timesteps) + bool(n_samples) < 2:
                msg = "Two of n_channels, n_timesteps, and n_channels must be provided"
                raise Exception(msg)
            if n_timesteps and n_channels:
                self.additional_info = {"channels": n_channels, "time": n_timesteps}
            elif n_timesteps and n_samples:
                self.additional_info = {"time": n_timesteps, "samples": n_samples}
            else:
                self.additional_info = {"channels": n_channels, "samples": n_samples}
        else:
            if n_channels is None and n_samples is None:
                msg = "One of n_channels or n_samples must be provided"
                raise Exception(msg)
            self.additional_info = {"channels": n_channels} if n_channels else {"samples": n_samples}

    def apply(self, img, **params):
        if self.time_dim:
            rearranged = rearrange(
                img,
                "(samples time channels) height width -> samples channels time height width",
                **self.additional_info,
            )
        else:
            rearranged = rearrange(
                img, "(samples channels) height width -> samples channels height width", **self.additional_info
            )
        return rearranged

    def get_transform_init_args_names(self):
        return ("n_timesteps", "n_channels")


class Rearrange(ImageOnlyTransform):
    """
    Rearrange is a generic image transformation that reshapes an input tensor using a custom einops pattern.

    This transform allows flexible reordering of tensor dimensions based on the provided pattern and arguments.
    """

    def __init__(
        self, rearrange: str, rearrange_args: dict[str, int] | None = None, always_apply: bool = True, p: float = 1
    ):
        """
        Initialize the Rearrange transform.

        Args:
            rearrange (str): The einops rearrangement pattern to apply.
            rearrange_args (dict[str, int] | None): Additional arguments for the rearrangement pattern.
            always_apply (bool): Whether to always apply this transform. Default is True.
            p (float): The probability of applying the transform. Default is 1.
        """
        super().__init__(always_apply, p)
        self.rearrange = rearrange
        self.vars = rearrange_args if rearrange_args else {}

    def apply(self, img, **params):
        return rearrange(img, self.rearrange, **self.vars)

    def get_transform_init_args_names(self):
        return "rearrange"


class SelectBands(ImageOnlyTransform):
    """
    SelectBands is an image transformation that selects a subset of bands (channels) from an input image.

    This transform uses specified band indices to filter and output only the desired channels from the image tensor.
    """

    def __init__(self, band_indices: list[int]):
        """
        Initialize the SelectBands transform.

        Args:
            band_indices (list[int]): A list of indices specifying which bands to select.
        """
        super().__init__(True, 1)
        self.band_indices = band_indices

    def apply(self, img, **params):
        return img[..., self.band_indices]

    def get_transform_init_args_names(self):
        return "band_indices"


def default_non_image_transform(array):
    if array.dtype == float or array.dtype == int:
        return torch.from_numpy(array)
    else:
        return array


class MultimodalToTensor:
    def __init__(self, modalities):
        self.modalities = modalities

    def __call__(self, d):
        new_dict = {}
        for k, v in d.items():
            if not isinstance(v, np.ndarray):
                new_dict[k] = v
            else:
                if k in self.modalities and len(v.shape) >= 3:  # Assuming raster modalities with 3+ dimensions
                    if len(v.shape) <= 4:
                        v = np.moveaxis(v, -1, 0)  # C, H, W or C, T, H, W
                    elif len(v.shape) == 5:
                        v = np.moveaxis(v, -1, 1)  # B, C, T, H, W
                    else:
                        raise ValueError(f"Unexpected shape for {k}: {v.shape}")
                new_dict[k] = torch.from_numpy(v)
        return new_dict


class MultimodalTransforms:
    """
    MultimodalTransforms applies albumentations transforms to multiple image modalities.

    This class supports both shared transformations across modalities and separate transformations for each modality.
    It also handles non-image modalities by applying a specified non-image transform.
    """

    def __init__(
        self,
        transforms: dict | A.Compose,
        shared: bool = True,
        non_image_modalities: list[str] | None = None,
        non_image_transform: object | None = None,
    ):
        """
        Initialize the MultimodalTransforms.

        Args:
            transforms (dict or A.Compose): The transformation(s) to apply to the data.
            shared (bool): If True, the same transform is applied to all modalities; if False, separate transforms are used.
            non_image_modalities (list[str] | None): List of keys corresponding to non-image modalities.
            non_image_transform (object | None): A transform to apply to non-image modalities. If None, a default transform is used.
        """
        self.transforms = transforms
        self.shared = shared
        self.non_image_modalities = non_image_modalities if non_image_modalities is not None else []
        self.non_image_transform = non_image_transform or default_non_image_transform

    def __call__(self, data: dict):
        if self.shared:
            # albumentations requires a key 'image' and treats all other keys as additional targets
            # (+ don't select 'mask' as 'image')
            image_modality = [k for k in data.keys() if k not in self.non_image_modalities + ["mask"]][0]
            data["image"] = data.pop(image_modality)
            data = self.transforms(**data)
            data[image_modality] = data.pop("image")

            # Process sequence data which is ignored by albumentations as 'global_label'
            for modality in self.non_image_modalities:
                data[modality] = self.non_image_transform(data[modality])
        else:
            # Applies transformations for each modality separate
            for key, value in data.items():
                if key in self.transforms:
                    if key in self.non_image_modalities:
                        raise NotImplementedError("Non image modalities not implemented.")
                    else:
                        data[key] = self.transforms[key](image=value)["image"]
                elif isinstance(value, np.ndarray):
                    data[key] = torch.from_numpy(value)
        return data


class AddConstantToLabels(nn.Module):
    def __init__(self, label_key="label", constant=1):
        super().__init__()
        self.label_key = label_key
        self.constant = constant

    def forward(self, sample):
        # Assumes sample is a dict with a 'label' key
        # and that sample['label'] is a torch.Tensor
        sample = sample.copy()  # To avoid modifying the original sample
        sample[self.label_key] = sample[self.label_key] + self.constant
        return sample
