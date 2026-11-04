# Copyright contributors to the Terratorch project


import logging

import terratorch.models.necks  # register necks
import terratorch.models.terracodec
from terratorch.models.clay1_5_model_factory import Clay1_5ModelFactory
from terratorch.models.clay_model_factory import ClayModelFactory
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
from terratorch.models.full_model_factory import FullModelFactory
from terratorch.models.generic_model_factory import GenericModelFactory
from terratorch.models.generic_unet_model_factory import GenericUnetModelFactory
from terratorch.models.object_detection_model_factory import ObjectDetectionModelFactory
from terratorch.models.prithvi_model_factory import PrithviModelFactory
from terratorch.models.smp_model_factory import SMPModelFactory
from terratorch.models.timm_model_factory import TimmModelFactory

try:
    granitewcx = True
    from terratorch.models.wxc_model_factory import WxCModelFactory
except ImportError:
    logging.getLogger("terratorch").debug("granitewxc not installed, please use pip install granitewxc")
    granitewcx = False

__all__ = (
    "Clay1_5ModelFactory",
    "ClayModelFactory",
    "EncoderDecoderFactory",
    "FullModelFactory",
    "GenericModelFactory",
    "GenericUnetModelFactory",
    "ObjectDetectionModelFactory",
    "PrithviModelFactory",
    "SMPModelFactory",
    "TimmModelFactory",
    "WxCModelFactory",
)

if granitewcx:
    __all__.__add__((WxCModelFactory,))
