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
| `olmoearth_v1_3_*` | `base` (`olmoearth-pretrain-minimal` >= 0.1.5) |

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

- Input: `[B, C, H, W]` or a time series `[B, C, T, H, W]`. `bands` lists the input channels per modality, in input
  order; it can be a subset, e.g. RGBN: band sets without any given band are masked as missing (v1.2 was trained with
  band dropout), missing bands of a partly given band set are zero. Without `bands`, the input is one `modality`
  (default `sentinel2_l2a`) with all bands in OlmoEarth's order. A dict `{modality: tensor}` works as well.
- Output: feature maps `[B, D, H / patch_size, W / patch_size]`, averaged over band sets and modalities.
  `modality_pooling: concat` stacks the modalities instead (`[B, n_modalities * D, ...]`).
- `temporal_pooling`: `mean` (default) or `concat` (time stacked into channels, needs `n_timestamps`).
- `fast_pass` (default `True`) skips the attention mask, which otherwise runs out of memory at small patch sizes.
- Do not use `backbone_use_temporal`: OlmoEarth encodes the time series itself.

## Tips

- **Normalization**: OlmoEarth expects raw values (Sentinel-2 L2A digital numbers, Sentinel-1 in dB) mapped from
  `mean ± 2·std` to `[0, 1]` with its pre-training statistics. Inputs standardized with dataset statistics look
  plausible but do not match. Get datamodule `means` and `stds` that reproduce it:

    ```python
    from terratorch.models.backbones.olmoearth import olmoearth_normalization

    means, stds = olmoearth_normalization({"sentinel2_l2a": ["B04", "B03", "B02", "B08"]})
    ```

- **Dates**: acquisition dates default to zeros. Pass them as `timestamps` `[B, T, 3]` (`[day, month, year]`, months
  0 to 11): add a `timestamps` entry to the batch (TerraTorch tasks pass extra batch entries to the backbone) or to a
  dict input.
- **Sequence length**: OlmoEarth was pre-trained with up to 12 time steps (`max_sequence_length`). Longer series
  run, but are outside the pre-training range.
- **Memory**: the encoder attends over `(H / patch_size)² · T · band sets` tokens (Sentinel-2 has 3 band sets,
  Sentinel-1 one). Halving the patch size quadruples the tokens; keep `fast_pass: True` for small patch sizes.
- **Time series**: with `temporal_pooling: concat`, a learned temporal pooling neck (`TemporalPooling`) can weight the
  time steps instead of averaging them.
- **Elevation**: `srtm` is accepted by `olmoearth-pretrain` and `olmoearth-pretrain-minimal` < 0.1, but it was only a
  decoding target during pre-training, so its encoder embedding is untrained.
- **Frozen encoder**: works with `freeze_backbone: True`.

See the [OlmoEarth repository](https://github.com/allenai/olmoearth_pretrain) for the pre-training data and
evaluation details.
