# Galileo

[Galileo](https://github.com/nasaharvest/galileo) ([paper](https://arxiv.org/abs/2502.09356)) is a multimodal,
multi-temporal foundation model by NASA Harvest. The encoder is included in TerraTorch, no extra package is needed.

## Model Versions

`galileo_nano`, `galileo_tiny`, `galileo_base`. `pretrained=True` downloads the weights from Hugging Face (default
`False`).

## Usage

=== "YAML"
    ```yaml
    model_args:
      backbone: galileo_tiny
      backbone_pretrained: True
      backbone_patch_size: 8
      backbone_bands: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, VV, VH, elevation]
      backbone_out_indices: [2, 5, 8, 11]  # for hierarchical decoders
      necks:
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      ...
    ```

=== "Python"
    ```python
    model_args = {
        "backbone": "galileo_tiny",
        "backbone_pretrained": True,
        "backbone_patch_size": 8,
        "backbone_bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "VV", "VH", "elevation"],
        "backbone_out_indices": [2, 5, 8, 11],
        "necks": [{"name": "LearnedInterpolateToPyramidal"}],
        "decoder": "UNetDecoder",
        ...
    }
    ```

- Input: `[B, C, H, W]` or a time series `[B, C, T, H, W]`. `bands` names the channels with Galileo's band names;
  they are sorted into Galileo's input families (e.g. `elevation` into the space family). Bands without a slot (here
  `B1`, `B9`) are ignored, missing band groups are masked. With `B4` and `B8`, NDVI is computed
  (`compute_ndvi`, default `True`). Without `bands`, the input holds Galileo's 13 space-time bands in order.
- Output: feature maps `[B, D, H / patch_size, W / patch_size]`, averaged over band groups.
- `patch_size` (default 4) can be any size with the pre-trained weights: the 8 px patch embeddings are resized to it
  FlexiViT-style (pseudo-inverse resize of the kernel). So set `backbone_patch_size` directly; `backbone_patch_resize`
  is not needed.
- `temporal_pooling`: `mean` (default) or `concat` (time stacked into channels, needs `n_timestamps`).
- Do not use `backbone_use_temporal`: Galileo encodes the time series itself.

## Tips

- **Normalization**: Galileo expects raw values (Sentinel-2 digital numbers, Sentinel-1 in dB, elevation in meters)
  mapped from `mean ± 2·std` to `[0, 1]` with its pre-training statistics
  ([`config/normalization.json`](https://github.com/nasaharvest/galileo/blob/main/config/normalization.json)). Inputs
  standardized with dataset statistics look plausible but do not match. Get datamodule `means` and `stds` that
  reproduce it:

    ```python
    from terratorch.models.backbones.galileo import galileo_normalization

    means, stds = galileo_normalization(["B2", "B3", "B4", "B8", "VV", "VH", "elevation"])
    ```

- **NDVI**: computed from `B4` and `B8` after undoing this normalization, with negative reflectances clamped to 0, so
  it requires inputs normalized as above. Otherwise, set `compute_ndvi: False` or pass an `NDVI` channel (not
  normalized).
- **Dates**: months default to 0. Pass them as `months` `[B, T]` (0 to 11): add a `months` entry to the batch
  (TerraTorch tasks pass extra batch entries to the backbone) or to a dict input.
- **Memory**: each band group is a separate token, so the encoder attends over about
  `(H / patch_size)² · (7 · T + 3)` tokens. At 256 px, `patch_size: 16` keeps this manageable; smaller patch sizes
  quadruple the tokens per halving.
- **Time series**: with `temporal_pooling: concat`, a learned temporal pooling neck (`TemporalPooling`) can weight the
  time steps instead of averaging them.
- **Frozen encoder**: works with `freeze_backbone: True`.

See the [Galileo repository](https://github.com/nasaharvest/galileo) for the band definitions and pre-training data.
