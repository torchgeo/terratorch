"""Static int8 (QDQ) quantization of the fcos NPU export.

The EdgeTPU is int8-only, so fp32/fp16 cannot run on it at all -- and since the Pixel exposes no
GPU NNAPI driver, an unquantized model has nowhere to go but the CPU. uint8 QDQ is the format
ORT's NNAPI EP consumes. Calibration uses real validation tiles at raw 0-255, matching the
terratorch datamodule (constant_scale=1, no mean/std normalisation).
"""
import argparse
import glob

import numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process


class TileReader(CalibrationDataReader):
    def __init__(self, files, size, input_name):
        self.input_name = input_name
        self.size = size
        self.files = files
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        f = self.files[self.i]
        self.i += 1
        im = Image.open(f).convert("RGB").resize((self.size, self.size))
        a = np.asarray(im).astype(np.float32).transpose(2, 0, 1)[None]  # raw 0-255
        return {self.input_name: a}

    def rewind(self):
        self.i = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/fcos_npu_fp32.onnx")
    ap.add_argument("--out", default="/tmp/fcos_npu_int8.onnx")
    ap.add_argument("--calib-dir", default="/home/romeokienzler/Downloads/trainod/val_images")
    ap.add_argument("--num-calib", type=int, default=64)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument(
        "--per-channel",
        action="store_true",
        help="Per-channel weight quantization. Recovers the accuracy that per-tensor loses on "
        "EfficientNet-Lite's depthwise convolutions, but the EdgeTPU cannot consume per-channel "
        "weights: measured on a Pixel 9, NNAPI drops from 15 ms to 96 ms (slower than the 45 ms "
        "CPU path). Use it when you intend to run on CPU, not on the NPU.",
    )
    ap.add_argument(
        "--quantize-tail",
        action="store_true",
        help="Also quantize the head's output Reshape/Concat/Transpose nodes. Measured on a "
        "Pixel 9 this makes ORT's NNAPI EP reject the model with ORT_INVALID_ARGUMENT "
        "'unsupported quantized type ... type: 2', so it is off by default.",
    )
    args = ap.parse_args()

    prepped = args.model.replace(".onnx", "_prep.onnx")
    quant_pre_process(args.model, prepped, skip_symbolic_shape=False)
    print(f"pre-processed -> {prepped}")

    import onnx
    m = onnx.load(prepped, load_external_data=False)
    input_name = m.graph.input[0].name

    files = sorted(glob.glob(f"{args.calib_dir}/*.png"))[: args.num_calib]
    print(f"calibrating on {len(files)} tiles, input '{input_name}'")

    # Keep the head's output Reshape/Concat/Transpose in float. Quantizing them trips ORT's NNAPI
    # EP ("unsupported quantized type ... type: 2") and the model falls back to CPU entirely.
    # Leaving them float costs ~6 ms on CPU but is what lets 615 of 687 nodes reach the EdgeTPU:
    # 20 ms via NNAPI vs 70 ms on CPU.
    exclude = []
    if not args.quantize_tail:
        exclude = [
            n.name
            for n in m.graph.node
            if n.op_type in ("Reshape", "Concat", "Transpose") and "/head/" in (n.name or "")
        ]
        print(f"keeping {len(exclude)} head tail nodes in float")

    quantize_static(
        prepped,
        args.out,
        TileReader(files, args.size, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        per_channel=args.per_channel,
        reduce_range=False,
        nodes_to_exclude=exclude,
    )
    import os
    print(f"wrote {args.out}  {os.path.getsize(args.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
