"""Export a trained fcos checkpoint's backbone+FPN+head to fp32 ONNX.

Reconstructs the ObjectDetectionTask from the checkpoint (so model_args, including
head_norm=batchnorm, come from the checkpoint's own hparams) and exports only the dense network.
Box decoding and NMS stay on the CPU.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from terratorch.tasks import ObjectDetectionTask


class DenseNet(nn.Module):
    """backbone + FPN + head. Output order is the head dict sorted by key:
    bbox_ctrness, bbox_regression, cls_logits."""

    def __init__(self, tvm):
        super().__init__()
        self.backbone = tvm.backbone
        self.head = tvm.head

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, dict):
            feats = list(feats.values())
        out = self.head(feats)
        if isinstance(out, dict):
            return tuple(out[k] for k in sorted(out))
        return out


def load_task(ckpt_path):
    task = ObjectDetectionTask.load_from_checkpoint(ckpt_path, map_location="cpu")
    task.eval()
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="/tmp/fcos_trained_fp32.onnx")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()

    task = load_task(args.ckpt)
    tvm = task.model.torchvision_model
    net = DenseNet(tvm).eval()

    x = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        outs = net(x)
    print("output order (sorted head keys): bbox_ctrness, bbox_regression, cls_logits")
    print("output shapes:", [tuple(o.shape) for o in outs])

    with torch.no_grad():
        torch.onnx.export(
            net, (x,), args.out,
            input_names=["images"],
            output_names=["bbox_ctrness", "bbox_regression", "cls_logits"],
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
