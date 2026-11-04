# Copyright contributors to the Terratorch project

# import so they get registered
import terratorch.models.backbones.clay_v1
import terratorch.models.backbones.dinov3
import terratorch.models.backbones.dofa_vit
import terratorch.models.backbones.gfm_swin
import terratorch.models.backbones.identity_backbone
import terratorch.models.backbones.mmearth_convnextv2
import terratorch.models.backbones.prithvi_swin
import terratorch.models.backbones.prithvi_vit
import terratorch.models.backbones.scalemae
import terratorch.models.backbones.terramind
import terratorch.models.backbones.torchgeo_resnet
import terratorch.models.backbones.torchgeo_swin_satlas
import terratorch.models.backbones.torchgeo_vit
from terratorch.models.backbones.unet import UNet

try:
    import terratorch.models.backbones.heliofm_register
except ImportError:
    pass
