# DINOv3

[DINOv3](https://github.com/facebookresearch/dinov3) (Meta, [paper](https://arxiv.org/abs/2508.10104)) is a family
of self-supervised vision foundation models. TerraTorch loads the architectures with `torch.hub` (internet access on
the first build) and the weights from a local checkpoint.

---

## Model Versions

Available model names:

```text
dinov3_vits16
dinov3_vits16plus
dinov3_vitb16
dinov3_vitl16
dinov3_vitl16plus
dinov3_vith16plus
dinov3_vit7b16
dinov3_convnext_tiny
dinov3_convnext_small
dinov3_convnext_base
dinov3_convnext_large
```

The weights are distributed by Meta after accepting the license
([download](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)). Pass the downloaded file as
`ckpt_path`; with `ckpt_path=None` the model is randomly initialized. Besides the web-image (LVD-1689M) models, Meta
provides ViT-L checkpoints pre-trained on satellite imagery (SAT-493M), which load into `dinov3_vitl16`.

The models were pre-trained on RGB images.

## Usage

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build("dinov3_vitl16", ckpt_path="path/to/dinov3_vitl16_pretrain_sat493m.pth")
features = model(x)  # x: [B, 3, H, W], RGB
```

### Multispectral input

With `bands`, the first layer (the ViT patch embedding or the ConvNeXt stem) is widened to `len(bands)` channels. The
pretrained RGB kernels are copied to the red, green and blue bands, named `RED`/`GREEN`/`BLUE` or `B04`/`B03`/`B02`
(also `B4`/`B3`/`B2`); the other channels are randomly initialized. Without any RGB band, the first layer is fully
random, with a warning.

```python
model = BACKBONE_REGISTRY.build(
    "dinov3_convnext_tiny", ckpt_path="path/to/dinov3_convnext_tiny_pretrain_lvd1689m.pth",
    bands=["B2", "B3", "B4", "B8", "B11", "B12"],
)
features = model(x)  # x: [B, 6, H, W]
```

### Model Output

The backbone returns a list of token tensors `[B, N, D]`, with the class token first (`return_cls_token=True`,
default):

- ViT: one tensor per transformer block, all at stride 16.
- ConvNeXt: one tensor per stage, at strides 4, 8, 16 and 32, with embedding sizes `[96, 192, 384, 768]` (tiny,
  small) and `[128, 256, 512, 1024]` (base), `[192, 384, 768, 1536]` (large).

`norm` (default `True`) applies the model's output norms; `norm=False` returns the raw block or stage outputs.

### Fine-tuning

=== "YAML"
    ```yaml
    model:
      class_path: terratorch.tasks.SemanticSegmentationTask
      init_args:
        model_factory: EncoderDecoderFactory
        model_args:
          backbone: dinov3_convnext_tiny
          backbone_ckpt_path: path/to/dinov3_convnext_tiny_pretrain_lvd1689m.pth
          backbone_bands: [B2, B3, B4, B8, B11, B12]
          necks:
            - name: ReshapeTokensToImage  # each stage to a 2D grid; ConvNeXt is already hierarchical
          decoder: UNetDecoder
          decoder_channels: [256, 128, 64, 32]
          ...
    ```

=== "Python"
    ```python
    task = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": "dinov3_convnext_tiny",
            "backbone_ckpt_path": "path/to/dinov3_convnext_tiny_pretrain_lvd1689m.pth",
            "backbone_bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "necks": [{"name": "ReshapeTokensToImage"}],
            "decoder": "UNetDecoder",
            "decoder_channels": [256, 128, 64, 32],
            ...
        },
        ...
    )
    ```

For ViTs with a hierarchical decoder, select four blocks and upsample them:

```yaml
      necks:
        - name: SelectIndices
          indices: [5, 11, 17, 23]  # ViT-L; [2, 5, 8, 11] for ViT-S/B
        - name: ReshapeTokensToImage
        - name: LearnedInterpolateToPyramidal
```

For multi-temporal inputs, wrap the backbone with the [TemporalWrapper](temporal_wrapper.md)
(`backbone_use_temporal: True`).
