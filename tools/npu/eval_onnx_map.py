"""Evaluate a trained fcos checkpoint on the test set, natively or with ONNX in the loop.

torchvision's FCOS.forward uses backbone features for exactly three things: head input, anchor
generation, and per-level sizes. So the network can be swapped for an ONNX session while keeping
torchvision's own box decoding and NMS -- making the int8 number directly comparable to the fp32
baseline rather than a reimplementation.

  eval_onnx_map.py --ckpt best.ckpt                    # PyTorch fp32 baseline
  eval_onnx_map.py --ckpt best.ckpt --onnx int8.onnx   # ONNX fp32 or int8 QDQ
"""
import argparse
import glob
import json
import os
import time
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from terratorch.tasks import ObjectDetectionTask
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Matches the datamodule: constant_scale=1 and ToTensorV2, i.e. raw 0-255 values, no mean/std.
# label_offset=-1 turns the 1-indexed labels in the JSON into 0-indexed class ids.
LABEL_OFFSET = -1


class OnnxTrunk(nn.Module):
    """Stands in for the backbone: runs ONNX, stashes head outputs, returns shape-only features."""

    def __init__(self, onnx_path, channels, threads):
        super().__init__()
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.channels = channels
        self.stash = None

    def forward(self, x):
        outs = self.sess.run(None, {self.input_name: x.detach().cpu().numpy()})
        by_name = dict(zip(self.out_names, outs))
        self.stash = {k: torch.from_numpy(by_name[k]) for k in
                      ("bbox_ctrness", "bbox_regression", "cls_logits")}
        n, _, h, w = x.shape
        return OrderedDict(  # strides for SelectIndices[1,2,3,4]
            (str(i), torch.zeros(n, self.channels, h // s, w // s))
            for i, s in enumerate((4, 8, 16, 32))
        )


class StashHead(nn.Module):
    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk

    def forward(self, features):
        return self.trunk.stash


def load_split(img_dir, label_dir, limit=0):
    items = []
    for img_path in sorted(glob.glob(f"{img_dir}/*.png")):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{stem}.json")
        if not os.path.exists(label_path):
            continue
        items.append((img_path, label_path))
        if limit and len(items) >= limit:
            break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--img-dir", default="/home/romeokienzler/Downloads/trainod/test_images")
    ap.add_argument("--label-dir", default="/home/romeokienzler/Downloads/trainod/test_labels")
    args = ap.parse_args()

    task = ObjectDetectionTask.load_from_checkpoint(args.ckpt, map_location="cpu")
    task.eval()
    tvm = task.model.torchvision_model

    if args.onnx:
        trunk = OnnxTrunk(args.onnx, args.channels, args.threads)
        tvm.backbone = trunk
        tvm.head = StashHead(trunk)
        label = f"ONNX {os.path.basename(args.onnx)}"
    else:
        label = "PyTorch fp32"

    items = load_split(args.img_dir, args.label_dir, args.limit)
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    t0 = time.perf_counter()
    with torch.no_grad():
        for img_path, label_path in items:
            im = Image.open(img_path).convert("RGB")
            x = torch.from_numpy(np.asarray(im).astype(np.float32).transpose(2, 0, 1))
            with open(label_path) as f:
                lab = json.load(f)
            boxes = torch.tensor(lab.get("boxes", []), dtype=torch.float32).reshape(-1, 4)
            labels = torch.tensor(
                [c + LABEL_OFFSET for c in lab.get("labels", [])], dtype=torch.int64
            )
            preds = tvm([x])
            metric.update(preds, [{"boxes": boxes, "labels": labels}])
    elapsed = time.perf_counter() - t0

    res = metric.compute()
    print(f"\n=== {label} | {len(items)} test images | {elapsed:.1f}s total ===")
    for k in ("map", "map_50", "map_75", "mar_100"):
        print(f"  {k:8s} {float(res[k]):.4f}")


if __name__ == "__main__":
    main()
