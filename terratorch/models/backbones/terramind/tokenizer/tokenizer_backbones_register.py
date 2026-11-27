# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings

import torch
from huggingface_hub import hf_hub_download

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

logger = logging.getLogger("terramind")

try:
    from .vqvae_backbone import VQBackbone

    vqvae_available = True
    import_error = None
except Exception as e:
    logger.debug(f"Could not import TerraMind tokenizer due to ImportError({e})")
    vqvae_available = False
    import_error = e


# Model definitions
__all__ = [
    "terramind_v1_tokenizer_dem",
    "terramind_v1_tokenizer_lulc",
    "terramind_v1_tokenizer_ndvi",
    "terramind_v1_tokenizer_s1grd",
    "terramind_v1_tokenizer_s1rtc",
    "terramind_v1_tokenizer_s2l2a",
]

pretrained_weights = {
    "terramind_v1_tokenizer_s2l2a": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S2L2A",
        "hf_hub_revision": "5a3f80664e0e701232c9526ca21eb7442a40f4be",
        "hf_hub_filename": "TerraMind_Tokenizer_S2L2A.pt",
    },
    "terramind_v1_tokenizer_s1rtc": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1RTC",
        "hf_hub_revision": "ea54d732d904c804b01e04cb5d85a462598bb9d0",
        "hf_hub_filename": "TerraMind_Tokenizer_S1RTC.pt",
    },
    "terramind_v1_tokenizer_s1grd": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1GRD",
        "hf_hub_revision": "38937fd35012c2dcf81987b400cec725b5eb0466",
        "hf_hub_filename": "TerraMind_Tokenizer_S1GRD.pt",
    },
    "terramind_v1_tokenizer_dem": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM",
        "hf_hub_revision": "aba0a5650a19eb82283e52ce59320c38c41f1e56",
        "hf_hub_filename": "TerraMind_Tokenizer_DEM.pt",
    },
    "terramind_v1_tokenizer_lulc": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-LULC",
        "hf_hub_revision": "af6b7c73a36994e25df156e793901969efe88ec9",
        "hf_hub_filename": "TerraMind_Tokenizer_LULC.pt",
    },
    "terramind_v1_tokenizer_ndvi": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-NDVI",
        "hf_hub_revision": "e1bd3b4d18023a7438221dbd6953e000cdc2dea5",
        "hf_hub_filename": "TerraMind_Tokenizer_NDVI.pt",
    },
}


def checkpoint_filter_fn(state_dict) -> dict:
    """Manually filter pre-trained weights for tokenizer backbone to enable strict weight loading."""

    for k in list(state_dict.keys()):
        if k.startswith("decoder."):
            _ = state_dict.pop(k)

    return state_dict


def build_vqvae(variant: str = None, pretrained: bool = False, ckpt_path: str | None = None, **kwargs):

    if not vqvae_available:
        warnings.warn(
            "Cannot import VQBackbone from terramind. \nMake sure to install `pip install diffusers==0.30.0`."
        )
        raise import_error

    model = VQBackbone(**kwargs)

    if ckpt_path is not None:
        # Load model from checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint_filter_fn(state_dict)
        loaded_keys = model.load_state_dict(state_dict, strict=False)
        if loaded_keys.missing_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")

    elif pretrained:
        # Load model from Hugging Face
        state_dict_file = hf_hub_download(
            repo_id=pretrained_weights[variant]["hf_hub_id"],
            filename=pretrained_weights[variant]["hf_hub_filename"],
            revision=pretrained_weights[variant].get("hf_hub_revision"),
        )
        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint_filter_fn(state_dict)
        model.load_state_dict(state_dict, strict=True)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_tokenizer_s2l2a(**kwargs):
    """
    Backbone from the S2L2A Tokenizer for TerraMind v1.
    """
    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_s2l2a",
        image_size=256,
        n_channels=12,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=True,
        **kwargs,
    )

    return tokenizer


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_tokenizer_s1rtc(**kwargs):
    """
    Backbone from the S1RTC Tokenizer for TerraMind v1.
    """

    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_s1rtc",
        image_size=256,
        n_channels=2,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=True,
        **kwargs,
    )

    return tokenizer


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_tokenizer_s1grd(**kwargs):
    """
    Backbone from the S1GRD Tokenizer for TerraMind v1.
    """

    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_s1grd",
        image_size=256,
        n_channels=2,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=True,
        **kwargs,
    )

    return tokenizer


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_tokenizer_dem(**kwargs):
    """
    Backbone from the DEM Tokenizer for TerraMind v1.
    """

    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_dem",
        image_size=256,
        n_channels=1,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=True,
        **kwargs,
    )

    return tokenizer


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_tokenizer_lulc(**kwargs):
    """
    Backbone from the LULC Tokenizer for TerraMind v1.
    """

    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_lulc",
        image_size=256,
        n_channels=10,
        encoder_type="vit_b_enc",
        decoder_type="vit_b_dec",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        quant_type="fsq",
        codebook_size="7-5-5-5-5",
        latent_dim=5,
        **kwargs,
    )

    return tokenizer


@TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_tokenizer_ndvi(**kwargs):
    """
    Backbone from the NDVI Tokenizer for TerraMind v1.
    """

    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_ndvi",
        image_size=256,
        n_channels=1,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=True,
        **kwargs,
    )

    return tokenizer
