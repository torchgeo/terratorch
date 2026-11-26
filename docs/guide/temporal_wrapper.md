# TemporalWrapper

The `TemporalWrapper` enables any TerraTorch backbone to process temporal input data and defines how features are aggregated over time. 

```python
class TemporalWrapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooling: Literal["keep", "concat", "mean", "max", "diff"] = "mean",
        n_timestamps: Optional[int] = None,
        subset_lengths: Optional[list[int]] = None,
    )

    def forward(self, 
            x: torch.Tensor | dict[str, torch.Tensor]
            ) -> list[torch.Tensor | dict[str, torch.Tensor]]:
```

## Functionality 
### Pooling modes
Select the temporal aggregation with the `pooling` parameter:

- `"keep"`: Preserve per-timestep outputs and return a temporal stack.
- `"concat"`: Concatenate features from all timesteps along the channel/feature dimension. Additional `n_timestamps` required.  
- `"mean"`: Average features across timesteps.  
- `"max"`: Element-wise maximum across timesteps. 
- `"diff"`: Compute the difference between two temporal elements, eg two timesteps (`t0 − t1`). If `subset_lengths` is provided, difference is computed between means of the defined subsets. 
- `"weighted"`: Learned weighted sum over time, `nn.Linear(T, 1)` shared across channels (T + 1 parameters per layer). Additional `n_timestamps` required.
- `"project"`: Learned projection of all timesteps and channels, a 1x1 `Conv2d(T*C, C)` (T·C² + C parameters per layer). Additional `n_timestamps` required.

`weighted` and `project` stay trainable with `freeze_backbone`.

!!! warning
    TerraTorch necks and decoders expect 4D inputs. Use a temporal aggregation that returns 4D (`mean`, `max`, `diff` or `concat`) for TerraTorch fine-tunings.

### Additional parameters

- `subset_lengths` (list[int], optional):  
  If set, performs two-step temporal aggregation:  
  (1) split timesteps into defined subsets (e.g., `[2, 3]` → first 2 and last 3 timesteps), average within each;  
  (2) apply the selected `pooling` across the resulting subset means.  
  Lengths must match the total number of timesteps.  
  For `pooling="diff"`, exactly two subsets are required.

### Inputs
TemporalWrapper expects 5D input data; depending on the wrapped backbone, provide either: 
- Tensor: `[B, C, T, H, W]`
- Multimodal dict: `{modality: [B, C_mod, T, H, W]}`

Each timestep is forwarded independently through the backbone. The resulting features are stacked and then either returned or aggregated along the temporal axis.

### Outputs
Backbones may return a list/tuple of layer outputs or a dictionary mapping modalities to such outputs. In all cases, TemporalWrapper applies temporal aggregation consistently:
- List/Tuple (multi-scale): Aggregate over time for each layer output independently.
- Dict (multimodal): Aggregate the time dimension independently per modality and per layer, preserving keys.


## Usage
### Wrap backbone
Initialize a backbone and pass it to the TemporalWrapper:

```python
    from terratorch.registry import BACKBONE_REGISTRY
    from terratorch.models.utils import TemporalWrapper

    # Build any TerraTorch backbone
    backbone = BACKBONE_REGISTRY.build("terramind_v1_base", modalities=["S2L2A"],pretrained=True)

    # Wrap it for temporal inputs
    temporal_backbone = TemporalWrapper(backbone=backbone, pooling="mean")

    # Forward with a temporal tensor x: [B, C, T, H, W]
    features = temporal_backbone(x)
```

### In Encoder–Decoder pipelines 
Use the wrapped model wherever a backbone is expected (e.g., in TerraTorch tasks):


=== "YAML"
    ```yaml
    model:
      class_path: terratorch.tasks.SemanticSegmentationTask
      init_args:
        model_factory: EncoderDecoderFactory
        model_args:
          backbone: prithvi_eo_v2_300_tl  # Select backbone
          backbone_pretrained: True  # Add backbone params
          backbone_use_temporal: True # Activate temporal wrapper
          backbone_temporal_pooling: "mean"  # Add params with prefix `backbone_temporal_` 
          ...
    ```

=== "Python"
    ```python
    import terratorch
    from terratorch.registry import BACKBONE_REGISTRY
    from terratorch.models.utils import TemporalWrapper
    
    # Option 1: Build the backbone manually and pass the nn.Module as backbone (easier debugging)     
    backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_300_tl", pretrained=True)
    temporal_backbone = TemporalWrapper(backbone, pooling="mean")
    
    task = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": temporal_backbone,
            ...
        },
        ...
    )
    
    # Option 2: Pass the options directly to the EncoderDecoderFactory
    task = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": "prithvi_eo_v2_300_tl",
            "backbone_pretrained": True,
            "backbone_use_temporal": True,   # Activate temporal wrapper
            "backbone_temporal_pooling": "mean"  # Pass arguments with prefix `backbone_temporal_`
            ...
        },
        ...
    )
    ```
 
!!! note
    For a TemporalWrapper backbone with `pooling='concat'`, set `n_timestamps` so dimensions (e.g. backbone output channels) are known at build time:
    
    ```python
    temporal_backbone = TemporalWrapper(backbone, pooling="concat", n_timestamps=6)
    ```

## TemporalPooling neck

Pools features with the timesteps concatenated in channels, `[B, T*C, H, W]` -> `[B, C, H, W]`, e.g. from Prithvi
(`ReshapeTokensToImage` with `effective_time_dim`) or a TemporalWrapper with `pooling: concat`. It supports the same
pooling modes and `subset_lengths`; `dim` (default 1) sets the concatenated dimension.

=== "YAML"
    ```yaml
    model_args:
      backbone: prithvi_eo_v2_300_tl
      backbone_pretrained: True
      backbone_num_frames: 4
      necks:
        - name: SelectIndices
          indices: [5, 11, 17, 23]
        - name: ReshapeTokensToImage
          effective_time_dim: 4  # -> [B, 4 * D, H', W']
        - name: TemporalPooling
          n_timestamps: 4
          pooling: weighted  # -> [B, D, H', W']
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      ...
    ```

=== "Python"
    ```python
    model_args={
        "backbone": "prithvi_eo_v2_300_tl",
        "backbone_pretrained": True,
        "backbone_num_frames": 4,
        "necks": [
            {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
            {"name": "ReshapeTokensToImage", "effective_time_dim": 4},
            {"name": "TemporalPooling", "n_timestamps": 4, "pooling": "weighted"},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder": "UNetDecoder",
        ...
    }
    ```
