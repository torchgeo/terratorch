# OlmoEarth

[OlmoEarth](https://github.com/allenai/olmoearth_pretrain) ([paper](https://arxiv.org/abs/2511.13655)) is a
multimodal, multi-temporal foundation model by Allen AI. Install `terratorch[olmoearth]` (`olmoearth-pretrain`) or
the lighter `olmoearth-pretrain-minimal`; both work.

## Model Versions

| Version | Sizes |
|---|---|
| `olmoearth_v1_*` | `nano`, `tiny`, `base`, `large` |
| `olmoearth_v1_1_*` | `nano`, `tiny`, `base` |
| `olmoearth_v1_2_*` | `nano`, `tiny`, `small`, `base` |

`pretrained=True` downloads the weights from Hugging Face (default `False`). OlmoEarth encodes `sentinel2_l2a`,
`sentinel1` and `landsat` time series.

## Usage

=== "YAML"
    ```yaml
    model_args:
      backbone: olmoearth_v1_2_tiny
      backbone_pretrained: True
      backbone_patch_size: 8
      backbone_bands:  # input channels per modality, in input order
        sentinel2_l2a: [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]
        sentinel1: [vv, vh]
      backbone_out_indices: [2, 5, 8, 11]  # for hierarchical decoders
      necks:
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      ...
    ```

=== "Python"
    ```python
    model_args = {
        "backbone": "olmoearth_v1_2_tiny",
        "backbone_pretrained": True,
        "backbone_patch_size": 8,
        "backbone_bands": {
            "sentinel2_l2a": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
            "sentinel1": ["vv", "vh"],
        },
        "backbone_out_indices": [2, 5, 8, 11],
        "necks": [{"name": "LearnedInterpolateToPyramidal"}],
        "decoder": "UNetDecoder",
        ...
    }
    ```

- Input: `[B, C, H, W]` or a time series `[B, C, T, H, W]`. Without `bands`, the input is one `modality` (default
  `sentinel2_l2a`) in OlmoEarth's band order. A dict `{modality: tensor, "timestamps": ...}` works as well.
- Output: feature maps `[B, D, H / patch_size, W / patch_size]`, averaged over band sets and modalities.
- `temporal_pooling`: `mean` (default) or `concat` (time stacked into channels, needs `n_timestamps`).
- `fast_pass` (default `True`) skips the attention mask, which otherwise runs out of memory at small patch sizes.
- Do not use `backbone_use_temporal`: OlmoEarth encodes the time series itself.
