# reference torchgeo https://torchgeo.readthedocs.io/en/stable/_modules/torchgeo/models/vit.html

import logging
from collections.abc import Callable
from functools import partial
from typing import List

import huggingface_hub
import torch
from torch import nn
from torchgeo.models import ViTSmall16_Weights, vit_small_patch16_224
from torchvision.models._api import Weights, WeightsEnum

from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


class ViTEncoderWrapper(nn.Module):
    """
    A wrapper for ViT models from torchgeo to return only the forward pass of the encoder
    Attributes:
        satlas_model (VisionTransformer): The instantiated dofa model
        weights
    Methods:
        forward(x: List[torch.Tensor], wavelengths: list[float]) -> torch.Tensor:
            Forward pass for embeddings with specified indices.
    """

    def __init__(self, vit_model, vit_meta, weights=None, out_indices=None) -> None:
        """
        Args:
            dofa_model (DOFA): The decoder module to be wrapped.
            weights ()
        """
        super().__init__()
        self.vit_model = vit_model
        self.weights = weights
        self.out_channels = [x["num_chs"] for x in self.vit_model.feature_info]
        self.vit_meta = vit_meta

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return self.vit_model.forward_intermediates(x, intermediates_only=True)


look_up_table = {
    "B01": "COASTAL_AEROSOL",
    "B02": "BLUE",
    "B03": "GREEN",
    "B04": "RED",
    "B05": "RED_EDGE_1",
    "B06": "RED_EDGE_2",
    "B07": "RED_EDGE_3",
    "B08": "NIR_BROAD",
    "B8A": "NIR_NARROW",
    "B09": "WATER_VAPOR",
    "B10": "CIRRUS",
    "B11": "SWIR_1",
    "B12": "SWIR_2",

    "B1": "COASTAL_AEROSOL",
    "B2": "BLUE",
    "B3": "GREEN",
    "B4": "RED",
    "B5": "RED_EDGE_1",
    "B6": "RED_EDGE_2",
    "B7": "RED_EDGE_3",
    "B8": "NIR_BROAD",
    "B8a": "NIR_NARROW",
    "B9": "WATER_VAPOR",

    "VV": "VV",
    "VH": "VH",
    "R": "RED",
    "G": "GREEN",
    "B": "BLUE",
}


def get_pretrained_bands(model_bands):

    model_bands = [look_up_table[x.split(".")[-1]] for x in model_bands]

    return model_bands


vit_s_meta = {"patch_size": 16, "embed_dim": 384, "depth": 12, "num_heads": 6}


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_vit_small_patch16_224_landsat_tm_toa_moco(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.LANDSAT_TM_TOA_MOCO,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_vit_small_patch16_224_landsat_tm_toa_simclr(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.LANDSAT_TM_TOA_SIMCLR,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_vit_small_patch16_224_landsat_etm_toa_moco(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.LANDSAT_ETM_TOA_MOCO,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_vit_small_patch16_224_landsat_etm_toa_simclr(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.LANDSAT_ETM_TOA_SIMCLR,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_vit_small_patch16_224_landsat_etm_sr_moco(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.LANDSAT_ETM_SR_MOCO,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_vit_small_patch16_224_landsat_etm_sr_simclr(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.LANDSAT_ETM_SR_SIMCLR,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_vit_small_patch16_224_landsat_oli_tirs_toa_simclr(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.LANDSAT_OLI_TIRS_TOA_SIMCLR,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_vit_small_patch16_224_landsat_oli_sr_moco(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.LANDSAT_OLI_SR_MOCO,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eol_vit_small_patch16_224_landsat_oli_sr_simclr(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.LANDSAT_OLI_SR_SIMCLR,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_vit_small_patch16_224_sentinel2_all_dino(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.SENTINEL2_ALL_DINO,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def ssl4eos12_vit_small_patch16_224_sentinel2_all_moco(
    model_bands,
    pretrained=False,
    ckpt_data: str | None = None,
    weights: Weights | None = ViTSmall16_Weights.SENTINEL2_ALL_MOCO,
    out_indices: list | None = None,
    **kwargs,
):
    """
    Args:
        model_bands (list[str]): A list containing the names for the bands expected by the model.
        pretrained (bool): The model is already pretrained (weights are available and can be restored) or not.
        ckpt_data (str | None): Path for a checkpoint containing the model weights.
    Returns:
        ViTEncoderWrapper
    """

    if "in_chans" not in kwargs:
        kwargs["in_chans"] = len(model_bands)
    model = vit_small_patch16_224(**kwargs)
    if pretrained:
        model = load_vit_weights(model, model_bands, ckpt_data, weights)
    return ViTEncoderWrapper(model, vit_s_meta, weights, out_indices)


#### to add build model and load weights
def load_vit_weights(
    model: nn.Module,
    model_bands,
    ckpt_data: str,
    weights: Weights,
    input_size: int = 224,
    custom_weight_proj: str = "patch_embed.proj.weight",
) -> nn.Module:

    pretrained_bands = (
        get_pretrained_bands(weights.meta["bands"])
        if "bands" in weights.meta
        else ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    )

    print("Loading weights")
    if ckpt_data is not None:
        if ckpt_data.find("https://hf.co/") > -1:
            repo_id = ckpt_data.split("/resolve/", maxsplit=1)[0].replace("https://hf.co/", "")
            filename = ckpt_data.rsplit("/", maxsplit=1)[-1]
            ckpt_data = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)
        state_dict = model.state_dict()

        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logging.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        checkpoint_model = select_patch_embed_weights(
            checkpoint_model, model, pretrained_bands, model_bands, custom_weight_proj
        )
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        logging.info(msg)
    elif weights is not None:
        checkpoint_model = weights.get_state_dict(progress=True, weights_only=True)
        state_dict = model.state_dict()

        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logging.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        checkpoint_model = select_patch_embed_weights(
            checkpoint_model, model, pretrained_bands, model_bands, custom_weight_proj
        )
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
        assert set(missing_keys) <= {"head.weight", "head.bias"}
        assert not unexpected_keys

    return model
