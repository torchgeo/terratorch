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
  `B1`, `B9`) are ignored, missing band groups are masked. Without `bands`, the input holds Galileo's 13 space-time
  bands in order.
- Output: feature maps `[B, D, H / patch_size, W / patch_size]`, averaged over band groups.
- `temporal_pooling`: `mean` (default) or `concat` (time stacked into channels, needs `n_timestamps`).
- `months` (`[B, T]`, zero-indexed) can be passed to the forward call; defaults to 0.
- Do not use `backbone_use_temporal`: Galileo encodes the time series itself.
