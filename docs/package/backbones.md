# Backbones

## Built-in Backbones

:::terratorch.models.backbones.terramind.model.terramind_vit.TerraMindViT
    options:
        toc_label: "TerraMind"

:::terratorch.models.backbones.prithvi_mae.PrithviViT
    options:
        toc_label: "Prithvi"

:::terratorch.models.backbones.swin_encoder_decoder.MMSegSwinTransformer
    options:
        toc_label: "Swin"

:::terratorch.models.backbones.unet.UNet
    options:
        toc_label: "U-Net"

:::terratorch.models.backbones.mmearth_convnextv2.ConvNeXtV2
    options:
        toc_label: "MMEarth ConvNeXt"

:::terratorch.models.backbones.dofa_vit.DOFAEncoderWrapper
    options:
        toc_label: "DOFA"

:::terratorch.models.backbones.satmae.SatMAEEncoder
    options:
        toc_label: "SatMAE"

:::terratorch.models.backbones.clay_v1.embedder
    options:
        toc_label: "Clay v1"

## APIs for External Models

!!! tip
    You find a detailed overview of all models in the [TorchGeo documentation](https://torchgeo.readthedocs.io/en/latest/api/models.html). 

:::terratorch.models.backbones.torchgeo_vit
    options:
        toc_label: "TorchGeo ViT models"

:::terratorch.models.backbones.torchgeo_resnet
    options:
        toc_label: "TorchGeo ResNet models"

:::terratorch.models.backbones.torchgeo_swin_satlas
    options:
        toc_label: "TorchGeo Swin Satlas"

:::terratorch.models.backbones.heliofm_register.heliofm_backbone_surya
    options:
        toc_label: "Surya"

<!--
### Timm

You can use any model from `timm` as a backbone. 

!!! tip
    List all available models with `timm.list_models` or filter by name using wildcards:
    
    ```python
    import timm
    timm.list_models('vit*')
    ```

::: timm.list_models
    options:
        heading_level: 4
        show_source: false
-->
