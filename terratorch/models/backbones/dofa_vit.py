# reference torchgeo https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/models/dofa.html#DOFA
import logging
import math
from collections.abc import Callable
from functools import partial
from typing import List

import huggingface_hub
import torch
import torch.nn.functional as F
from torch import nn
from torchgeo.models import dofa
from torchvision.models._api import Weights, WeightsEnum

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

waves_list = {
    "COASTAL_AEROSOL": 0.44,
    "BLUE": 0.49,
    "GREEN": 0.56,
    "RED": 0.665,
    "RED_EDGE_1": 0.705,
    "RED_EDGE_2": 0.74,
    "RED_EDGE_3": 0.783,
    "NIR_BROAD": 0.832,
    "NIR_NARROW": 0.864,
    "WATER_VAPOR": 0.945,
    "CIRRUS": 1.373,
    "SWIR_1": 1.61,
    "SWIR_2": 2.20,
    "THERMAL_INFRARED_1": 10.90,
    "THERMAL_INFRARED_12": 12.00,
    "VV": 5.405,
    "VH": 5.405,
    "ASC_VV": 5.405,
    "ASC_VH": 5.405,
    "DSC_VV": 5.405,
    "DSC_VH": 5.405,
    "VV-VH": 5.405,
}


def resize(
    input: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    warning: bool = True,
) -> torch.Tensor:
    """Resize input tensor with alignment warning check.

    Args:
        input: Input tensor of shape [B, C, H, W]
        size: Target output size (H, W)
        scale_factor: Multiplier for spatial size
        mode: Interpolation mode ('bilinear', 'bicubic'.)
        align_corners: If True, aligns corners for non-nearest modes
        warning: If True, warns about potential alignment issues

    Returns:
        Resized tensor of shape [B, C, H_new, W_new]
    """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
    """Resize pos_embed weights.
    Resize pos_embed using bicubic interpolate method.
    Args:
        pos_embed (torch.Tensor): Position embedding weights.
        input_shpae (tuple): Tuple for (downsampled input image height,
            downsampled input image width).
        pos_shape (tuple): The resolution of downsampled origin training
            image.
        mode (str): Algorithm used for upsampling:
            ``'bilinear'`` | ``'bicubic'`` . Default: ``'bilinear'``
    Return:
        torch.Tensor: The resized pos_embed of shape [B, L_new, C]
    """
    assert pos_embed.ndim == 3, "shape of pos_embed must be [B, L, C]"
    pos_h, pos_w = pos_shape
    cls_token_weight = pos_embed[:, 0]
    pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
    pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
    pos_embed_weight = resize(pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
    cls_token_weight = cls_token_weight.unsqueeze(1)
    pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
    pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
    return pos_embed


class DOFAEncoderWrapper(nn.Module):
    """
    A wrapper for DOFA models from torchgeo to return only the forward pass of the encoder
    Attributes:
        dofa_model (DOFA): The instantiated dofa model
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, dofa_model, wavelengths, weights=None, out_indices=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.dofa_model = dofa_model
        self.weights = weights
        self.wavelengths = wavelengths

        self.out_indices = out_indices if out_indices else [-1]
        self.out_channels = [self.dofa_model.patch_embed.embed_dim] * len(self.out_indices)

    def forward(self, x: list[torch.Tensor], **kwargs) -> torch.Tensor:
        wavelist = torch.tensor(self.wavelengths, device=x.device).float()

        x, _ = self.dofa_model.patch_embed(x, wavelist)
        x = x + self.dofa_model.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.dofa_model.cls_token + self.dofa_model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outs = []
        # apply Transformer blocks
        for i, block in enumerate(self.dofa_model.blocks):
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
            elif (i == (len(self.dofa_model.blocks) - 1)) & (-1 in self.out_indices):
                outs.append(x)

        return tuple(outs)


def get_wavelenghts(model_bands: list[str]) -> list[float]:
    """Extract wavelength values for given spectral bands.

    Args:
        model_bands: List of band names (e.g., ['RED', 'NIR', 'SWIR_1'])

    Returns:
        List of corresponding wavelength values in micrometers
    """
    wavelengths = [waves_list[x.split(".")[-1]] for x in model_bands]
    return wavelengths


@TERRATORCH_BACKBONE_REGISTRY.register
def dofa_small_patch16_224(
    model_bands,
    input_size=224,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = None,
    out_indices: list | None = None,
    pos_interpolation_mode: str = "bilinear",
    **kwargs,
):
    model = dofa.dofa_small_patch16_224(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, pos_interpolation_mode, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)

    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def dofa_base_patch16_224(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = dofa.DOFABase16_Weights.DOFA_MAE,
    out_indices: list | None = None,
    pos_interpolation_mode: str = "bilinear",
    **kwargs,
):
    model = dofa.dofa_base_patch16_224(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, pos_interpolation_mode, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)

    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def dofa_large_patch16_224(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = dofa.DOFALarge16_Weights.DOFA_MAE,
    out_indices: list | None = None,
    pos_interpolation_mode: str = "bilinear",
    **kwargs,
):
    model = dofa.dofa_large_patch16_224(**kwargs)
    input_size = kwargs["img_size"] if "img_size" in kwargs else 224
    if pretrained:
        model = load_dofa_weights(model, pos_interpolation_mode, ckpt_data, weights, input_size)
    wavelengths = get_wavelenghts(model_bands)

    return DOFAEncoderWrapper(model, wavelengths, weights, out_indices)


def load_dofa_weights(
    model: nn.Module,
    mode: str,
    ckpt_data: str | None = None,
    weights: Weights | None = None,
    input_size: int = 224,
    patch_size: int = 16,
) -> nn.Module:
    state_dict = model.state_dict()
    print("Loading weights")
    if ckpt_data is not None:
        if ckpt_data.find("https://hf.co/") > -1:
            repo_id = ckpt_data.split("/resolve/")[0].replace("https://hf.co/", "")
            filename = ckpt_data.split("/")[-1]
            ckpt_data = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)

        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logging.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        if input_size != 224:
            if "pos_embed" in checkpoint_model and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape:
                logging.info("Resizing pos_embed from pretrained checkpoint")
                h, w = input_size, input_size
                pos_size = int(math.sqrt(checkpoint_model["pos_embed"].shape[1] - 1))
                checkpoint_model["pos_embed"] = resize_pos_embed(
                    pos_embed=checkpoint_model["pos_embed"],
                    input_shpae=(h // patch_size, w // patch_size),
                    pos_shape=(pos_size, pos_size),
                    mode=mode,
                )

        msg = model.load_state_dict(checkpoint_model, strict=False)

        logging.info(msg)
    elif weights is not None:
        checkpoint_model = weights.get_state_dict(progress=True, weights_only=True)
        allowed_missing_keys = {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}
        if input_size != 224:
            if "pos_embed" in checkpoint_model and checkpoint_model["pos_embed"].shape != state_dict["pos_embed"].shape:
                logging.info("Removing key pos_embed from pretrained checkpoint")
                del checkpoint_model["pos_embed"]
                allowed_missing_keys.add("pos_embed")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
        logging.info("Weights loaded.")
        # Both fc_norm and head are generated dynamically
        assert set(missing_keys) <= allowed_missing_keys
        assert not unexpected_keys
    else:
        print("No weights to load.")

    return model
