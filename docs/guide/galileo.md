# Galileo

[Galileo](https://github.com/nasaharvest/galileo) (NASA Harvest, [paper](https://arxiv.org/abs/2502.09356)) is a
multimodal, multi-temporal foundation model for Earth observation. Its encoder is included in TerraTorch
(`terratorch/models/backbones/galileo.py`, MIT licensed), so no extra package is needed.

---

## Model Versions

| Name | Depth | Embedding | Params |
|---|---|---|---|
| `galileo_nano` | 4 | 128 | 1.0M |
| `galileo_tiny` | 12 | 192 | 5.7M |
| `galileo_base` | 12 | 768 | 87M |

Weights are downloaded from the Hugging Face Hub (`nasaharvest/galileo`) with `pretrained=True` (default `False`).

Galileo groups its inputs into four families, each split into band groups that are tokenized separately:

| Family | Shape | Bands |
|---|---|---|
| space-time | `[B, H, W, T, 13]` | `VV`, `VH`, `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`, `NDVI` |
| space | `[B, H, W, 16]` | `elevation`, `slope` (SRTM), Dynamic World classes (`DW_*`), WorldCover crops (`WC_*`) |
| time | `[B, T, 6]` | ERA5, TerraClimate, VIIRS |
| static | `[B, 18]` | LandScan, location, static Dynamic World / WorldCover |

Families and band groups without data are masked, as in Galileo, so the model treats them as absent.

## Usage

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build("galileo_tiny", pretrained=True, patch_size=8)
features = model(x)  # x: [B, 13, H, W] or [B, 13, T, H, W], space-time bands in Galileo's order
```

### Model Input

The backbone accepts:

- a tensor `[B, C, H, W]` or a time series `[B, C, T, H, W]` with the 13 space-time bands in Galileo's order;
- a tensor whose channels are named by `bands` (Galileo band names, case-insensitive). Space-time bands are reordered,
  space bands such as `elevation` go to the space family (averaged over time). Bands without a slot in Galileo, e.g.
  Sentinel-2 `B1` and `B9`, are ignored with a warning. Band groups that receive no band are masked, missing bands of a
  partly given group are set to zero;
- a dict of the native families: `space_time` (required), `space`, `time`, `static` and `months`.

```python
model = BACKBONE_REGISTRY.build(
    "galileo_tiny", pretrained=True,
    bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "VV", "VH", "elevation"],
)
features = model(x)  # x: [B, 15, T, H, W]
```

Acquisition months are passed as `months` `[B, T]` (zero-indexed), as the forward argument or the `"months"` dict
entry. They default to 0.

Other parameters:

- `patch_size` (default 4): other patch sizes than Galileo's base patch size (8) are handled FlexiViT-style, by
  resizing the input. Smaller patches give more tokens and use more memory.
- `temporal_pooling`: `"mean"` (default) averages the time steps; `"concat"` stacks them into channels,
  `[B, T * D, H', W']`, and needs `n_timestamps`. Combine `concat` with a temporal pooling neck to learn the
  aggregation over time.
- `out_indices`: encoder blocks whose outputs are returned, for multi-scale decoders.

The output averages the band groups of the space-time tokens. It is a list of feature maps
`[B, D, H / patch_size, W / patch_size]`, one per output level.

### Fine-tuning

The backbone returns feature maps, so it needs no `ReshapeTokensToImage` neck. For hierarchical decoders such as
UNet, select four blocks and upsample them:

=== "YAML"
    ```yaml
    model:
      class_path: terratorch.tasks.SemanticSegmentationTask
      init_args:
        model_factory: EncoderDecoderFactory
        model_args:
          backbone: galileo_tiny
          backbone_pretrained: True
          backbone_patch_size: 16
          backbone_bands: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, VV, VH, elevation]
          backbone_out_indices: [2, 5, 8, 11]  # depth 12; [0, 1, 2, 3] for nano
          necks:
            - name: LearnedInterpolateToPyramidal
          decoder: UNetDecoder
          decoder_channels: [256, 128, 64, 32]
          ...
    ```

=== "Python"
    ```python
    task = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": "galileo_tiny",
            "backbone_pretrained": True,
            "backbone_patch_size": 16,
            "backbone_bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
                               "VV", "VH", "elevation"],
            "backbone_out_indices": [2, 5, 8, 11],
            "necks": [{"name": "LearnedInterpolateToPyramidal"}],
            "decoder": "UNetDecoder",
            "decoder_channels": [256, 128, 64, 32],
            ...
        },
        ...
    )
    ```

The input is `[B, 15, T, H, W]`; `B1` and `B9` are ignored and the time steps are averaged by default.

Do not set `backbone_use_temporal`: Galileo encodes the time series itself, while the
[TemporalWrapper](temporal_wrapper.md) would encode each time step separately.
