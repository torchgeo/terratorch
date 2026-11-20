# OlmoEarth

[OlmoEarth](https://github.com/allenai/olmoearth_pretrain) (Allen AI, [paper](https://arxiv.org/abs/2511.13655)) is a
spatio-temporal, multimodal foundation model for Earth observation. TerraTorch wraps its encoder as a backbone.

Install one of the two OlmoEarth packages. They ship the same model code; `olmoearth-pretrain-minimal` is used if
both are installed.

```sh
pip install terratorch[olmoearth]          # olmoearth-pretrain
pip install olmoearth-pretrain-minimal     # lighter loader, without training dependencies
```

---

## Model Versions

| Name | Depth | Embedding | Params |
|---|---|---|---|
| `olmoearth_v1_nano` | 4 | 128 | 1.4M |
| `olmoearth_v1_tiny` | 12 | 192 | 6.2M |
| `olmoearth_v1_base` | 12 | 768 | 89M |
| `olmoearth_v1_large` | 24 | 1024 | 308M |
| `olmoearth_v1_1_nano` | 4 | 128 | 1.7M |
| `olmoearth_v1_1_tiny` | 12 | 192 | 12.5M |
| `olmoearth_v1_1_base` | 12 | 768 | 114M |
| `olmoearth_v1_2_nano` | 4 | 128 | 1.7M |
| `olmoearth_v1_2_tiny` | 12 | 192 | 12.5M |
| `olmoearth_v1_2_small` | 12 | 384 | 36M |
| `olmoearth_v1_2_base` | 12 | 768 | 114M |

Weights are downloaded from the Hugging Face Hub with `pretrained=True` (default `False`). The architecture is read
from the checkpoint config in both cases, so `pretrained=False` gives the released model with random weights.

OlmoEarth encodes `sentinel2_l2a` (12 bands), `sentinel1` (`vv`, `vh`) and `landsat` (11 bands), each as a time
series. Each modality's bands are split into band sets that are tokenized separately.

!!! note
    `srtm` (elevation) is accepted by `olmoearth-pretrain` and `olmoearth-pretrain-minimal` < 0.1, but in the released
    checkpoints it was only used as a decoding target during pre-training, so its encoder embedding is untrained.

## Usage

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build("olmoearth_v1_2_tiny", pretrained=True, patch_size=8)
features = model(x)  # x: [B, 12, H, W] Sentinel-2 L2A in OlmoEarth's band order
```

### Model Input

The backbone accepts:

- a tensor `[B, C, H, W]` or a time series `[B, C, T, H, W]` of one modality (`modality`, default `sentinel2_l2a`),
  with the bands in OlmoEarth's order for that modality;
- a stacked multi-sensor tensor described by `bands`: the channels of each modality, in input order. They are split
  into OlmoEarth's modalities and reordered, so any band order works. Each modality needs all of its bands;
- a dict `{modality: tensor}`, optionally with `"timestamps"`.

```python
model = BACKBONE_REGISTRY.build(
    "olmoearth_v1_2_tiny", pretrained=True,
    bands={
        "sentinel2_l2a": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
        "sentinel1": ["vv", "vh"],
    },
)
features = model(x)  # x: [B, 14, T, H, W]
```

Acquisition dates are passed as `timestamps` `[B, T, 3]` (`[day, month, year]`, months zero-indexed), as the forward
argument or the `"timestamps"` dict entry. They default to zeros.

All modalities are encoded jointly. The output averages the band sets of each modality, then the modalities, so the
number of channels does not depend on the inputs.

Other parameters:

- `patch_size` (default 8): other patch sizes are handled FlexiViT-style, by resizing the input. Smaller patches
  give more tokens and use more memory.
- `input_res` (default 10): ground sampling distance in meters.
- `temporal_pooling`: `"mean"` (default) averages the time steps; `"concat"` stacks them into channels,
  `[B, T * D, H', W']`, and needs `n_timestamps`. Combine `concat` with a temporal pooling neck to learn the
  aggregation over time.
- `out_indices`: encoder blocks whose outputs are returned, for multi-scale decoders.
- `fast_pass` (default `True`): no attention mask in training, so that PyTorch can use memory-efficient attention.
  Without it, the mask is expanded to `[B, heads, N, N]`, which runs out of memory at small patch sizes. It does not
  change the outputs, since the wrapper never masks tokens.
- `ckpt_path`: load a local checkpoint instead of the Hugging Face weights.

### Fine-tuning

The backbone returns feature maps `[B, D, H / patch_size, W / patch_size]`, so it needs no
`ReshapeTokensToImage` neck. For hierarchical decoders such as UNet, select four blocks and upsample them:

=== "YAML"
    ```yaml
    model:
      class_path: terratorch.tasks.SemanticSegmentationTask
      init_args:
        model_factory: EncoderDecoderFactory
        model_args:
          backbone: olmoearth_v1_2_tiny
          backbone_pretrained: True
          backbone_patch_size: 16
          backbone_bands:
            sentinel2_l2a: [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]
            sentinel1: [vv, vh]
          backbone_out_indices: [2, 5, 8, 11]  # depth 12; [0, 1, 2, 3] for nano, [5, 11, 17, 23] for large
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
            "backbone": "olmoearth_v1_2_tiny",
            "backbone_pretrained": True,
            "backbone_patch_size": 16,
            "backbone_bands": {
                "sentinel2_l2a": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
                "sentinel1": ["vv", "vh"],
            },
            "backbone_out_indices": [2, 5, 8, 11],
            "necks": [{"name": "LearnedInterpolateToPyramidal"}],
            "decoder": "UNetDecoder",
            "decoder_channels": [256, 128, 64, 32],
            ...
        },
        ...
    )
    ```

The input is `[B, 14, T, H, W]`; the time steps are averaged by default (`temporal_pooling: mean`).

Do not set `backbone_use_temporal`: OlmoEarth encodes the time series itself, while the
[TemporalWrapper](temporal_wrapper.md) would encode each time step separately.
