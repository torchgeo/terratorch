# TerraMind

TerraMind is a multi-modal, generative foundation model build by IBM and ESA. 
It is fully integrated into TerraTorch and support standard fine-tuning, Thinking-in-Modalities (TiM), and generation tasks.

More information about the TerraMind models: [https://ibm.github.io/terramind/](https://ibm.github.io/terramind/)

If you encounter any issues, please create an issue in our [GitHub repo](https://github.com/IBM/terramind).

---

## Model Versions

TerraMind 1.0 Backbones (`BACKBONE_REGISTRY`)

```text
terramind_v1_tiny
terramind_v1_small
terramind_v1_base
terramind_v1_large
terramind_v1_tiny_tim
terramind_v1_small_tim
terramind_v1_base_tim
terramind_v1_large_tim
```
The tiny or small versions are recommended for local development, edge deployment, fast inference, or very simple downstream tasks. 
TerraMind base is best suited for evaluating or deploying most downstream tasks.
If performance is very important and compute is available, TerraMind large is expected to perform best.
Thinking-in-Modalities (TiM) models (see section below) can improve downstream task performance depending on the task and always needs to be validated.


TerraMind 1.0 Generative Models (`FULL_MODEL_REGISTRY`)

```text
terramind_v1_tiny_generate
terramind_v1_small_generate
terramind_v1_base_generate
terramind_v1_large_generate
```

Note that all generation models are using the same diffusion decoders. Therefore, TerraMind tiny and small are not much faster then the base or large versions, while the quality can be very different.

TerraMind 1.0 Tokenizers (`FULL_MODEL_REGISTRY`)

```text
terramind_v1_tokenizer_s2l2a
terramind_v1_tokenizer_s1rtc
terramind_v1_tokenizer_s1grd
terramind_v1_tokenizer_dem
terramind_v1_tokenizer_lulc
terramind_v1_tokenizer_ndvi
```

Raw input modalities supported by backbones: `S2L1C`, `S2L2A`, `RGB`, `S1GRD`, `S1RTC`, `DEM`, and `Coordinates`
(*Note: RGB patch embedding was pre-trained on Sentinel-2 RGB inputs \[0–255].*)

Tokenized input modalities (use with caution for fine-tuning): `LULC`, `NDVI`
See [New Modalities](#new-modalities) for alternatives.

Modalities usable as `tim_modalities` or `output_modalities`: `S2L2A`, `S1GRD`, `S1RTC`, `DEM`, `LULC`, `NDVI`, `Coordinates`

!!! info
    Coordinate support is added in `terratorch==1.1`. Install TerraTorch from a recent version or from the `main` branch.

??? quote "Experimental Models (v0.1)"
    TerraMind v0.1 models (TM-B-single in the paper):
    ```yaml
        terramind_v01_base
        terramind_v01_base_tim
        terramind_v01_base_generate
    ```
    These are experimental and not publicly released. It supports only `S2L2A` as raw input and `Captions` as input and output modality.

---

## Usage

### Fine-Tuning

Use `modalities` to define input types. You can specify them in `BACKBONE_FACTORY.build` for building the backbone or `model_args` when building the task-specific model.

TerraMind uses seperated tokens per modality. If you use multiple modalities, these tokens need to be merged for each patch embedding to be compatible with the decoders.  
By default, the encoder merges the embeddings across image modalities by averaging (`mean`). Another approach can be selected with `merge_method`:
`max` takes the maximum value over all image modality while `concat` concatenates the modalities along the embedding dimension which increases the decoder input.
In a custom python script, you can also use `dict` which returns a dictionary with all embeddings rather than a tensor, or `None` which keeps the modalities as seperated tokens.

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build(
    "terramind_v1_base", pretrained=True, modalities=["S2L2A", "S1GRD"],
    merge_method='concat' # mean(default), max, concat, dict, None
)
```


Use TerraMind as a backbone in TerraTorch's EncoderDecoderFactory: 
=== "YAML"
    ```yaml
    model:
      class_path: terratorch.tasks.SemanticSegmentationTask
      init_args:    
        model_factory: EncoderDecoderFactory
        model_args:
          backbone: terramind_v1_base
          backbone_pretrained: True
          backbone_modalities:
            - S2L2A
            - S1GRD
          backbone_merge_method: mean  # mean (default), max, concat
          ...
    ```

=== "Python"
    ```python
    task = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory", 
        model_args={
            "backbone": "terramind_v1_base",
            "backbone_pretrained": True,
            "backbone_modalities": ["S2L2A", "S1GRD"],
            "backbone_merge_method": "mean", # mean (default), max, concat
            ...
        },
        ...
    )
    ```

The backbone output is a list of tensors \[batch, token, embedding] (no CLS token) which need to be restructured depending on the decoder.
For hierarchical decoders (e.g., UNet), use the following necks:

=== "YAML"
    ```yaml
    model_args:    
      ...
      necks:
        - name: ReshapeTokensToImage  # Reshape 1D tokens to 2D grid 
          remove_cls_token: False
        - name: SelectIndices  # Select three intermediate layer outputs and the final one
          indices: [2, 5, 8, 11]  # Base model
        # indices: [5, 11, 17, 23]  # Large model
        - name: LearnedInterpolateToPyramidal  # Upscale outputs for hierarchical decoders
        ...
    ```

=== "Python"
    ```python
    model_args={
        ...        
        "necks": [
            {"name": "ReshapeTokensToImage", 
             "remove_cls_token": False}
            {"name": "SelectIndices", 
            "indices": [2, 5, 8, 11]}, # Base model
            # "indices": [5, 11, 17, 23]}, # Large model
            {"name": "LearnedInterpolateToPyramidal"}
        ]
        ...
    } 
    ```

You can find an example for fine-tuning TerraMind with multi-modal inputs in this [notebook](https://github.com/IBM/terramind/blob/main/notebooks/terramind_v1_base_sen1floods11.ipynb) and this [config](https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_sen1floods11.yaml) file.

Set `modality_drop_rate` to train TerraMind that supports multiple modalities but can handle inference on a subset (e.g., a single input). 
During training, modalities are randomly dropped according to the rate (e.g., with `0.1` each modality is dropped in 10% of all batches).

!!! info "Local checkpoint"
    You can load model weights from a local checkpoint by passing `ckpt_path` to `BACKBONE_REGISTRY.build()` or define `backbone_ckpt_path` in the model args.

### Model Input

The model expects the input as a dict `model({"mod1": input1, "mod2": input2, ...})` or as keyword args `model(mod1=input1, mod2=input2)` with inputs being torch tensors.
If you only use a single modality, you can pass the input direct as a tensor `model(input1)`.
In TerraTorch, single modality data modules such as the `GenericNonGeoSegmentationDataModule` us the latter option, while the `GenericMultiModalDataModule` loads the inputs as a dictionary by default.


```python
import torch
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build("terramind_v1_base", pretrained=True, modalities=["S2L2A", "S1GRD"])

s2_input: torch.Tensor = torch.rand(1, 12, 224, 224)  # [B, C, H, W]
s1_input: torch.Tensor = torch.rand(1, 2, 224, 224)  # [B, C, H, W]

# Input as dict
out = model({
    "S2L2A": s2_input,
    "S1GRD": s1_input,
})

# Input as kwargs
out = model(S2L2A=s2_input, S1GRD=s2_input)

# One modality (assume first defined modality)
out = model(s2_input)

# The output is a list of tensors from each transformer layer
len(out)  
# 12

out[-1].shape  # Output shape: [Batch, Tokens, Embedding]
# torch.Size([1, 196, 768])
```

### Patch Size

The pre-trained patch embeddings are 16x16, so `backbone_patch_size` other than 16 raises an error with
`backbone_pretrained: True` (without pre-trained weights, it sets the patch size of all patch embeddings).
To use a smaller effective patch size with the pre-trained weights, set `backbone_patch_resize` instead.
It upsamples the input so that one 16x16 patch covers `backbone_patch_resize` input pixels, FlexiViT-style,
and keeps the pre-trained patch embeddings unchanged. E.g., `backbone_patch_resize: 8` resizes a 256 px
input to 512 px, giving 32 x 32 tokens. Labels keep their resolution.

```yaml
backbone: terramind_v1_base
backbone_pretrained: True
backbone_modalities: [S2L2A, S1RTC]
backbone_patch_resize: 8
```

### Subset of Input Bands

Use the `bands` dict to select a subset of the pre-trained bands. Unlisted modalities expect all bands as inputs.

Here is an example that uses the Sentinel-2 embeddings for Landsat-8 data. `S2L1C` still needs to be part of the modality name so that TerraTorch knows which patch embeddings to load (the name can be quite flexible, e.g., upper or lower case works).

=== "YAML"
    ```yaml
    backbone_modalities:
      - S2L1C_L8
      - S1GRD
    backbone_bands:
      S2L1C_L8:  # Modality name has to match backbone_modalities
        - COASTAL_AEROSOL
        - BLUE
        - GREEN
        - RED
        - NIR_NARROW
        - SWIR_1
        - SWIR_2
        - PANCHROMATIC  # New band
        - CIRRUS
        - THERMAL_1  # New band
        - THERMAL_2  # New band
    ```

=== "Python"
    ```python
    model_args={
        ...
        "backbone_modalities": ["S2L1C_L8", "S1GRD"],
        "backbone_bands": {
            "S2L1C_L8": [
                "COASTAL_AEROSOL",
                "BLUE",
                "GREEN",
                "RED",
                "NIR_NARROW",
                "SWIR_1",
                "SWIR_2",
                "PANCHROMATIC",  # New band
                "CIRRUS",
                "THERMAL_1",  # New band
                "THERMAL_2",  # New band
            ] 
        },
        ...
    }
    ```

=== "Backbone registry"
    ```python
    model = BACKBONE_REGISTRY.build("terramind_v1_base", pretrained=True, 
                                    modalities=["S2L2A", "S1GRD"],
                                    bands={
                                        "S2L1C_L8": [
                                        "COASTAL_AEROSOL",
                                        "BLUE",
                                        "GREEN",
                                        "RED",
                                        "NIR_NARROW",
                                        "SWIR_1",
                                        "SWIR_2",
                                        "PANCHROMATIC",  # New band
                                        "CIRRUS",
                                        "THERMAL_1",  # New band
                                        "THERMAL_2",  # New band
                                    ]}                                
                                    )
    ```

??? into "List of pre-trained bands"
    The name of the pre-trained bands are [here](https://github.com/IBM/terratorch/blob/53768e684a50e3f7e37d654f499dcccb4373940b/terratorch/models/backbones/terramind/model/terramind_register.py#L77) specified:
    ```yaml
    S2L1C / S2L2A:
      - COASTAL_AEROSOL
      - BLUE
      - GREEN
      - RED
      - RED_EDGE_1
      - RED_EDGE_2
      - RED_EDGE_3
      - NIR_BROAD
      - NIR_NARROW
      - WATER_VAPOR
      - CIRRUS  # Only in S2L1C
      - SWIR_1
      - SWIR_2 
    S1GRD / S1RTC:
      - VV
      - VH 
    RGB:
      - BLUE
      - GREEN
      - RED 
    DEM: 
      - DEM
    ```

### New modalities

You might want to use input modalities which are not used with raw inputs during pre-training. 
Therefore, you can define a new patch embedding by providing a dict as an input in the modality list, which specifies the new name and the number of input channels.
You can also use it if you want to fine-tune the model with NDVI or LULC data, which would otherwise use the TerraMind tokenizers, which increases the model size quite a lot.
Here is an example that reuses the S-2 patch embedding but initalizes a new patch emebdding for NDVI data and a completly new modality:  

=== "YAML"
    ```yaml
    backbone_modalities:
      - S2L2A
      - NDVI: 1
      - PLANET: 6
    ```

=== "Python"
    ```python
    model_args={
        ...
        "backbone_modalities": [
            "S2L2A",
            {"NDVI": 1},
            {"PLANET": 6}
        ],
        ...
    }
    ```

=== "Backbone registry"
    ```python
    model = BACKBONE_REGISTRY.build("terramind_v1_base", pretrained=True, 
                                    modalities=[
                                        "S2L2A", 
                                        {"NDVI": 1},
                                        {"PLANET": 6}
                                    ])
    ```

Note that in our experience it is always better to reuse a patch embedding, even with other satellites (e.g. using S-2 or RGB modalities for other optical sensors).
The model more quickly adapts to the new data rather than learning it from scratch.
The current implementation cannot reuse a specific patch embedding multiple times. However, you could use up to three optical modalities (using S2L1C, S2L2A, and RGB) and two SAR modalities (S1GRD and S1RTC). For example: 

=== "YAML"
    ```yaml
    backbone_modalities:
      - S2L2A
      - NDVI: 1
      - S2L1C_PLANET # Reuse S2L1C patch embedding
    backbone_bands:
      S2L1C_PLANET:
        - BLUE
        - GREEN
        - RED
        - ...
    ```
=== "Python"
    ```python
    model_args={
        ...
        "backbone_modalities": [
            "S2L2A",
            {"NDVI": 1},
            "S2L1C_PLANET"
        ],
        "backbone_bands": {
            "S2L1C_PLANET": [
                "BLUE",
                "GREEN",
                "RED",
                ...
            ] 
        ...
    }
    ```

=== "Backbone registry"
    ```python
    model = BACKBONE_REGISTRY.build("terramind_v1_base", pretrained=True, 
                                    modalities=[
                                        "S2L2A", 
                                        {"NDVI": 1},
                                        {"PLANET": 6}
                                    ],
                                    bands={
                                        "S2L1C_PLANET": [
                                        "BLUE",
                                        "GREEN",
                                        "RED",
                                        ...
                                    ]}                                
                                    )
    ```

To define a single new modality directly:

=== "YAML"
    ```yaml
    model_args:
      backbone: terramind_v1_base
      backbone_pretrained: True
      backbone_modalities: []
      backbone_in_chans: 3
    ```

=== "Python"
    ```python
    model_args={
        ...
        "backbone": "terramind_v1_base",
        "backbone_pretrained": True,
        "backbone_modalities": [],
        "backbone_in_chans": 3,
        ...
    },
    ```

=== "Backbone registry"
    ```python
    model = BACKBONE_REGISTRY.build("terramind_v1_base", pretrained=True, 
                                    modalities=[],
                                    in_chans=3,
    ```


This creates a random patch embedding named `image`, usable with a raw tensor or `{"image": tensor}` as model input.

### Inference

In order to run inference with multimodal data using the multimodal generic data module, you can pass a dict with your modalities as `predict_data_root`: 

```shell
terratorch predict -c "config_deploy.yaml" --ckpt_path "best-state_dict-epoch=09.ckpt" --predict_output_dir /data/ --data.init_args.predict_data_root '{"S1GRD": "/data/task_0", "S2L1C": "/data/task_0"}'
```

You can also add the `predict_data_root` directly in your config instead of using a CLI. If you have the tiff files of all modalities in a single folder like this example, make sure to set `image_grep` to identify the files per modality.

---

## Thinking-in-Modalities (TiM)

During fine-tuning or inference, TerraMind can pause for a moment, imagine a helpful but absent layer, append the imagined tokens to its own input sequence, and then lets the fine-tuned encoder continue to improve its own performance. 
Because the imagination lives in token space, the approach avoids the heavy diffusion decoding that full image synthesis would require. 
So, TerraMind can generate any missing modality as an intermediate step — an ability we call Thinking in Modalities (TiM). 
We refer to the [paper](https://arxiv.org/pdf/2504.11171) for details.

!!! warning "Important"
    TiM only works with fully pre-trained raw inputs (all bands, no `bands` parameter).
    The generator model is frozen and cannot adapt to unseen inputs such as subsets of pre-trained bands or new modalities.
    If this is the case for you, you cannot use the TiM models.

To use, suffix `_tim` to the model name and set `tim_modalities`.


=== "YAML"
    ```yaml
    model_args:
      backbone: terramind_v1_base_tim
      backbone_pretrained: True
      backbone_modalities:
        - S2L2A
      backbone_tim_modalities:
        - LULC
    ```

=== "Python"
    ```python
    model_args={
        ...
        "backbone": "terramind_v1_base",
        "backbone_pretrained": True,
        "backbone_modalities": ["S2L2A"],
        "backbone_tim_modalities": ["LULC"],
        ...
    },
    ```

=== "Backbone registry"
    ```python
    model = BACKBONE_REGISTRY.build(
        "terramind_v1_base_tim",
        pretrained=True,
        modalities=["S2L2A", "S1GRD"],
        tim_modalities=["LULC"]
    )
    ```

Here is a [TiM config](https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_tim_lulc_sen1floods11.yaml) example for fine-tuning. 

---

## Generation

Use `*_generate` models for any-to-any generation.

```python
model = BACKBONE_REGISTRY.build(
    'terramind_v1_base_generate',
    modalities=['S2L2A'],
    output_modalities=['S2L2A', 'S1GRD', 'LULC'],
    pretrained=True,
    standardize=True,
)
```

Use `standardize=True` to automatically apply correct scaling to the inputs and generations.
Alternatively, you can find the standardization values from pre-training [here](https://github.com/IBM/terratorch/blob/53768e684a50e3f7e37d654f499dcccb4373940b/terratorch/models/backbones/terramind/model/terramind_register.py#L130).
The model requires all pre-trained bands for the inputs.

Demos are provided in [terramind_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_generation.ipynb) and [any_to_any_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_any_to_any_generation.ipynb).
You can use TerraTorch's large-tile-inference to generate images of larger scenes, which is demonstrated in [large_tile_generation.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/large_tile_generation.ipynb).

---

## Tokenizers

Initialize tokenizers from `FULL_MODEL_REGISTRY`:

```python
from terratorch.registry import FULL_MODEL_REGISTRY

model = FULL_MODEL_REGISTRY.build('terramind_v1_tokenizer_s2l2a', pretrained=True)
```

Reconstruction example:

```python
with torch.no_grad():
    # Full reconstruction
    reconstruction = model(normalized_input)

    # Encode only
    _, _, tokens = model.encode(normalized_input)

    # Decode only
    reconstruction = model.decode_tokens(tokens)
```

We provide an example in [terramind_tokenizer_reconstruction.ipynb](https://github.com/IBM/terramind/blob/main/notebooks/terramind_tokenizer_reconstruction.ipynb).

