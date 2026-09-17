"""Which framework+backbone combinations can terratorch actually build for an NPU-targeted export?

EdgeTPU/NNAPI need int8-quantizable, statically-shaped, pure-conv graphs. That rules out the
DETR family (GridSample/LayerNorm) and the R-CNN family (RoIAlign + dynamic proposal counts),
leaving the dense single-stage heads: fcos and retinanet. The open question is the backbone.
"""
import warnings

warnings.filterwarnings("ignore")

import torch
from terratorch.models.object_detection_model_factory import ObjectDetectionModelFactory

# Candidates, cheapest/most NPU-friendly first. All are pure-CNN (no attention, no LayerNorm).
BACKBONES = [
    "timm_mobilenetv3_large_100",
    "timm_efficientnet_lite0",
    "timm_tf_efficientnetv2_b0",
    "timm_resnet18",
    "timm_resnet34",
    "timm_resnet50",
]

factory = ObjectDetectionModelFactory()

for framework in ("fcos", "retinanet"):
    for bb in BACKBONES:
        try:
            model = factory.build_model(
                task="object_detection",
                backbone=bb,
                framework=framework,
                num_classes=2,
                in_channels=3,
                backbone_pretrained=False,
                backbone_in_chans=3,
                framework_min_size=384,
                framework_max_size=384,
            )
            model.eval()
            n = sum(p.numel() for p in model.parameters())
            # what feature channels does the backbone hand the FPN?
            oc = getattr(model.torchvision_model.backbone, "channel_list", None) \
                if hasattr(model, "torchvision_model") else None
            print(f"OK    {framework:10s} {bb:32s} params={n/1e6:6.2f}M channels={oc}")
        except Exception as e:
            msg = str(e).replace("\n", " ")[:110]
            print(f"FAIL  {framework:10s} {bb:32s} {type(e).__name__}: {msg}")
