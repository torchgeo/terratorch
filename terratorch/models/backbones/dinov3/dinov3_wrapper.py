import warnings

import numpy as np
import torch
from torch import Tensor, nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

# DINOv3 is RGB-pretrained: its first conv's input channels are red, green, blue.
# Accepted names per channel: TerraTorch HLS band names and Sentinel-2 band codes.
RGB_BAND_NAMES = (("RED", "B04", "B4"), ("GREEN", "B03", "B3"), ("BLUE", "B02", "B2"))


class DinoV3Wrapper(nn.Module):
    def __init__(
        self,
        model: str,
        ckpt_path: str = None,
        return_cls_token: bool = True,
        bands: list[str] | None = None,
        norm: bool = True,
        **kwargs,
    ):
        """
        TerraTorch wrapper for DINO V3 models.

        Args:
        model: model name from the ones supported in https://github.com/facebookresearch/dinov3/blob/main/hubconf.py
        ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
        return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.
        bands: names of the input channels, for inputs other than RGB (e.g. multispectral). The first conv is widened
            to len(bands) channels; the pretrained kernels go to the red, green and blue bands (RED/GREEN/BLUE or
            B04/B03/B02), the other channels are randomly initialized. Defaults to RGB input.
        norm: whether to apply the model's output norms to the returned features.

        Returns:
            DinoV3Wrapper model

        """

        super().__init__()
        if ckpt_path is None:
            self.dinov3 = torch.hub.load("facebookresearch/dinov3", model, pretrained=False)
        else:
            self.dinov3 = torch.hub.load("facebookresearch/dinov3", model, weights=ckpt_path)

        self.out_channels = (
            [self.dinov3.embed_dim] * len(self.dinov3.blocks)
            if hasattr(self.dinov3, "blocks")
            else self.dinov3.embed_dims
        )
        self.output_indexes = list(np.arange(len(self.dinov3.blocks))) if hasattr(self.dinov3, "blocks") else 4
        self.return_cls_token = return_cls_token
        self.norm = norm
        if bands is not None:
            self._widen_first_conv([str(getattr(b, "value", b)).upper() for b in bands])

    def _widen_first_conv(self, bands: list[str]) -> None:
        """Replace the 3-channel stem (ConvNeXt) or patch embedding (ViT) by a len(bands)-channel one."""
        parent, name = (
            (self.dinov3.patch_embed, "proj")
            if hasattr(self.dinov3, "patch_embed")
            else (self.dinov3.downsample_layers[0], "0")
        )
        old = getattr(parent, name)
        new = nn.Conv2d(
            len(bands), old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None
        )
        rgb = [next((i for i, b in enumerate(bands) if b in names), None) for names in RGB_BAND_NAMES]
        if all(i is None for i in rgb):
            warnings.warn(
                f"DINOv3: no red, green or blue band in {bands}, the first conv is randomly initialized.", stacklevel=3
            )
        with torch.no_grad():
            for src, dst in enumerate(rgb):
                if dst is not None:
                    new.weight[:, dst] = old.weight[:, src]
            if old.bias is not None:
                new.bias.copy_(old.bias)
        setattr(parent, name, new)

    def forward(self, x: Tensor):
        """
        Forward pass for the model.

        Args:
            x: tensor of shape Bx3xHxW

        Returns:
            list of emebddings for all the intermediate layers blocks.
        """

        feats = self.dinov3.get_intermediate_layers(
            x, n=self.output_indexes, return_class_token=self.return_cls_token, norm=self.norm
        )

        if self.return_cls_token:
            # rearrange with classe token in front
            feats = [torch.cat([x[0], x[1].unsqueeze(1)], axis=1) for x in feats]

        return list(feats)


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vits16(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vits16 model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vits16 model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vits16", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vits16plus(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vits16plus model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vits16plus model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vits16plus", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vitb16(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vitb16 model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vitb16 model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vitb16", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vitl16plus(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vitb16 model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vitl16plus model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vitl16plus", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vitl16(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vitl16 model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vitl16 model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vitl16", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vith16plus(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vith16plus model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vith16plus model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vith16plus", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vit7b16(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vit7b16 model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vit7b16 model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vit7b16", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_convnext_base(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_convnext_base model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_convnext_base model

    """

    # Init model
    model = DinoV3Wrapper(
        model="dinov3_convnext_base", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs
    )

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_convnext_large(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_convnext_large model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_convnext_large model

    """

    # Init model
    model = DinoV3Wrapper(
        model="dinov3_convnext_large", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs
    )

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_convnext_small(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_convnext_small model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_convnext_small model

    """

    # Init model
    model = DinoV3Wrapper(
        model="dinov3_convnext_small", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs
    )

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_convnext_tiny(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_convnext_tiny model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_convnext_tiny model

    """

    # Init model
    model = DinoV3Wrapper(
        model="dinov3_convnext_tiny", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs
    )

    return model
