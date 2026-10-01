# SatMAE

TerraTorch provides the static multispectral `satmae_base` and `satmae_large`
encoders through the normal `EncoderDecoderFactory`. They require the fixed
native ten-band Sentinel-2 order and 96×96 inputs. Radiometric conversion and
geometry are dataset-transform responsibilities.

With `pretrained=True`, the official fMoW-Sentinel pretraining checkpoint is
downloaded from Zenodo record 7338613 and MD5-verified. Pass `ckpt_path` to
use a local copy. Checkpoints are not redistributed by TerraTorch; the Zenodo
record licenses them CC-BY-4.0. The implementation is a modified encoder-only
port of Apache-2.0 ExPLoRA commit `5b8cdcb704eead1b4cfe7ba1d6c870fb58ec8afd`.
