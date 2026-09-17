"""Export the NPU-delegable part of the fcos + EfficientNet-Lite0 detector to fp32 ONNX.

backbone -> FPN -> dense head only. Box decoding / score thresholding / NMS stay on the CPU.
Weights are whatever the factory produces (ImageNet backbone init, random head) -- this export
exists to test NNAPI/EdgeTPU operator coverage and latency, not accuracy.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from terratorch.models.object_detection_model_factory import ObjectDetectionModelFactory


class DenseNet(nn.Module):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--backbone", default="timm_efficientnet_lite0")
    ap.add_argument("--framework", default="fcos")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--out", default="/tmp/fcos_npu_fp32.onnx")
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()

    model = ObjectDetectionModelFactory().build_model(
        task="object_detection",
        backbone=args.backbone,
        framework=args.framework,
        num_classes=args.num_classes,
        in_channels=3,
        backbone_pretrained=args.pretrained,
        backbone_in_chans=3,
        head_norm="batchnorm",
        necks=[
            {"name": "SelectIndices", "indices": [1, 2, 3, 4]},
            {"name": "FeaturePyramidNetworkNeck", "out_channel": 128},
        ],
        framework_min_size=args.size,
        framework_max_size=args.size,
    )
    model.eval()
    net = DenseNet(model.torchvision_model).eval()

    x = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        outs = net(x)
    print("output shapes:", [tuple(o.shape) for o in outs])

    with torch.no_grad():
        torch.onnx.export(
            net, (x,), args.out,
            input_names=["images"],
            output_names=[f"out{i}" for i in range(len(outs))],
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
    params = sum(p.numel() for p in net.parameters())
    print(f"exported {args.out}  params={params/1e6:.2f}M")


if __name__ == "__main__":
    main()
