# DINOv3

[DINOv3](https://github.com/facebookresearch/dinov3) ([paper](https://arxiv.org/abs/2508.10104)) is a family of
self-supervised vision models by Meta, loaded via `torch.hub`.

## Model Versions

ViTs: `dinov3_vits16`, `dinov3_vits16plus`, `dinov3_vitb16`, `dinov3_vitl16`, `dinov3_vitl16plus`,
`dinov3_vith16plus`, `dinov3_vit7b16`. ConvNeXts: `dinov3_convnext_{tiny,small,base,large}`.

Weights are downloaded from [Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and passed as
`ckpt_path` (`None` for random weights). This includes ViT-L weights pre-trained on satellite images (SAT-493M).

## Usage

=== "YAML"
    ```yaml
    model_args:
      backbone: dinov3_convnext_tiny
      backbone_ckpt_path: path/to/dinov3_convnext_tiny_pretrain_lvd1689m.pth
      backbone_bands: [B2, B3, B4, B8, B11, B12]  # optional, default RGB
      necks:
        - name: ReshapeTokensToImage
      decoder: UNetDecoder
      ...
    ```

=== "Python"
    ```python
    model_args = {
        "backbone": "dinov3_convnext_tiny",
        "backbone_ckpt_path": "path/to/dinov3_convnext_tiny_pretrain_lvd1689m.pth",
        "backbone_bands": ["B2", "B3", "B4", "B8", "B11", "B12"],  # optional, default RGB
        "necks": [{"name": "ReshapeTokensToImage"}],
        "decoder": "UNetDecoder",
        ...
    }
    ```

- `bands`: widens the first layer to `len(bands)` channels. The pretrained RGB weights go to `RED`/`GREEN`/`BLUE`
  (or `B04`/`B03`/`B02`), the other channels are randomly initialized.
- Output: tokens `[B, N, D]` with the class token first, one per ViT block or ConvNeXt stage. For ViTs with a
  hierarchical decoder, add `SelectIndices` and `LearnedInterpolateToPyramidal` (as for Prithvi).
- `norm` (default `True`) applies the output norms.

## Tips

- **Normalization**: DINOv3 expects RGB in `[0, 1]`, standardized with the statistics of its pre-training data:
  ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]` for the web-image (LVD-1689M) models, mean
  `[0.430, 0.411, 0.296]` / std `[0.213, 0.156, 0.143]` for the satellite (SAT-493M) models.
- **Multispectral input**: only the RGB channels start from pretrained weights; all other bands are learned during
  fine-tuning.
- **Decoders**: ConvNeXts are hierarchical (strides 4 to 32) and work with a UNet decoder directly. ViTs output a
  single stride-16 scale and need `SelectIndices` + `ReshapeTokensToImage` + `LearnedInterpolateToPyramidal` for
  hierarchical decoders.
- **Time series**: use the [TemporalWrapper](temporal_wrapper.md) (`backbone_use_temporal: True`).
- **Offline**: the first build downloads the architecture code via `torch.hub`; afterwards it is loaded from the
  `torch.hub` cache.

See the [DINOv3 repository](https://github.com/facebookresearch/dinov3) for the checkpoints and their licenses.
