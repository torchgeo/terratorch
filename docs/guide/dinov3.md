# DINOv3

[DINOv3](https://github.com/facebookresearch/dinov3) ([paper](https://arxiv.org/abs/2508.10104)) is a family of
self-supervised vision models by Meta, loaded via `torch.hub`.

## Model Versions

ViTs: `dinov3_vits16`, `dinov3_vits16plus`, `dinov3_vitb16`, `dinov3_vitl16`, `dinov3_vitl16plus`,
`dinov3_vith16plus`, `dinov3_vit7b16`. ConvNeXts: `dinov3_convnext_{tiny,small,base,large}`.

Weights are downloaded from [Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and passed as
`ckpt_path` (`None` for random weights). This includes ViT-L weights pre-trained on satellite images (SAT-493M).

## Usage

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

- `bands`: widens the first layer to `len(bands)` channels. The pretrained RGB weights go to `RED`/`GREEN`/`BLUE`
  (or `B04`/`B03`/`B02`), the other channels are randomly initialized.
- Output: tokens `[B, N, D]` with the class token first, one per ViT block or ConvNeXt stage. For ViTs with a
  hierarchical decoder, add `SelectIndices` and `LearnedInterpolateToPyramidal` (as for Prithvi).
- `norm` (default `True`) applies the output norms.
