# NPU-targeted object detection (branch `odapp`)

Tooling for training an object detector in terratorch that runs on a phone NPU — specifically the
Pixel 9 / Tensor G4 EdgeTPU, reached through ONNX Runtime's NNAPI execution provider.

**The full handoff document, including all measured numbers, the working quantization recipe,
environment gotchas and outstanding work, lives in the companion Android repo:**

    /home/romeokienzler/gitco/odapp/status.md    → section "HANDOFF (2026-09-17)"

Read that first. Short version:

- RF-DETR (the previously deployed model) cannot run on NNAPI at all: GridSample, LayerNorm and
  ~850 dynamic-shape ops fragment it into 27 partitions, then NNAPI model building fails outright.
- **FCOS + `timm_efficientnet_lite0`** with `head_norm: batchnorm` exports to a graph with zero
  NPU-hostile operators. Trained on the trainod dataset it reaches **test map 0.9793 as int8**
  (RF-DETR: 0.9765) with 4.8M params, and runs in **13 ms on the EdgeTPU** vs ~2120 ms for RF-DETR.
- `head_norm` is a new option added to `ObjectDetectionModelFactory` on this branch. It must be set
  at training time, because BatchNorm needs learned running statistics.

## Scripts

| script | purpose |
|---|---|
| `export_trained_fcos.py` | trained checkpoint → fp32 ONNX (backbone+FPN+head only; no decode/NMS) |
| `quantize_fcos_int8.py` | fp32 ONNX → uint8 QDQ int8; defaults are the verified NPU-compatible recipe |
| `eval_onnx_map.py` | test-set mAP with ONNX in the loop, reusing torchvision's own decode + NMS |
| `export_fcos_npu.py` | untrained export, for operator-coverage and latency probing |
| `probe_od_backbones.py` | which framework/backbone combinations the factory can build |

All require `PYTHONPATH=/home/romeokienzler/gitco/terratorch.odapp` and
`/home/romeokienzler/tmp/.venv/bin/python`, because terratorch is also installed non-editable in
that venv and would otherwise shadow this branch.

## Configs

- `examples/object_detection/object_detection_trainod_fcos_npu.yaml` — produced the result above.
- `examples/object_detection/object_detection_trainod_fcos_npu_r18.yaml` — ResNet18 variant. Written
  when an undertrained checkpoint suggested EfficientNet-Lite's depthwise convolutions were breaking
  per-tensor int8. On the converged model that effect disappeared, so this was never trained and is
  probably unnecessary. Kept only in case per-tensor quantization becomes a problem on other data.
