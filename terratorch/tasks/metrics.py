import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


class BoundaryMeanIoU(Metric):
    """Boundary mIoU for multiclass segmentation.

    Boundary IoU (Cheng et al., 2021) per class: the IoU of the inner contour bands of prediction and target, i.e.
    the class pixels within d = ``thickness`` pixels of a class contour (mask minus its erosion by a (2d+1)^2 square).
    Intersections and unions are summed over the dataset, then averaged over classes with a non-empty union (macro)
    or pooled (micro).

    Differences from the reference implementation (https://github.com/bowenc0221/boundary-iou-api), for tiled
    remote sensing data:

    - d is fixed in pixels (default 4), not 2 % of the image diagonal: at a fixed ground sample distance this is a
      fixed ground distance, independent of the tile size.
    - The image border is not a contour: tile edges are arbitrary cuts, and bands along them would match trivially.
    - Pixels within d of an ``ignore_index`` target are left out, not only the ignored pixels: contours next to
      missing labels are artifacts of the missing labels.

    Metric based on https://arxiv.org/abs/2103.16562:
    Cheng, B., Girshick, R., Dollár, P., Berg, A. C., & Kirillov, A. (2021). Boundary IoU: Improving object-centric
    image segmentation evaluation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.

    Args:
        num_classes: Number of classes.
        thickness: Band width d in pixels, inside each class region.
        ignore_index: Target value excluded from the metric.
        average: ``"macro"`` or ``"micro"``.
        include_background: Whether class 0 is scored.
    """

    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        thickness: int = 4,
        ignore_index: int | None = None,
        average: str = "macro",  # "macro" or "micro"
        include_background: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if average not in {"macro", "micro"}:
            raise ValueError("average must be 'macro' or 'micro'")

        self.num_classes = num_classes
        self.thickness = thickness
        self.ignore_index = ignore_index
        self.average = average
        self.include_background = include_background

        # accumulators across batches
        self.add_state("intersections", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("unions", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    @staticmethod
    def _inner_band(mask: Tensor, width: int) -> Tensor:
        """Pixels of a (N, H, W) bool mask within `width` of its contour; the image border is not a contour."""
        # erosion as a min-pool; max_pool2d pads -x with -inf, so the outside never erodes the mask
        eroded = -F.max_pool2d(-mask.unsqueeze(1).float(), kernel_size=2 * width + 1, stride=1, padding=width)
        return mask & (eroded.squeeze(1) < 0.5)

    @torch.no_grad()
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        preds: (N, C, H, W) logits/probs OR (N, H, W) class indices
        target: (N, H, W) class indices
        """
        if preds.dim() == 4:
            # logits -> hard labels
            preds_idx = preds.argmax(dim=1)
        elif preds.dim() == 3:
            preds_idx = preds
        else:
            raise ValueError("preds must be (N,C,H,W) or (N,H,W)")

        if target.dim() != 3:
            raise ValueError("target must be (N,H,W)")

        width = self.thickness

        # leave out pixels within `width` of an ignored target
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            ignored = (target == self.ignore_index).unsqueeze(1).float()
            valid = F.max_pool2d(ignored, kernel_size=2 * width + 1, stride=1, padding=width).squeeze(1) < 0.5

        # optionally skip background class
        start_cls = 0 if self.include_background else 1

        for c in range(start_cls, self.num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            bnd_pred = self._inner_band(preds_idx == c, width) & valid
            bnd_target = self._inner_band(target == c, width) & valid

            self.intersections[c] += (bnd_pred & bnd_target).sum()
            self.unions[c] += (bnd_pred | bnd_target).sum()

    def compute(self) -> Tensor:
        eps = 1e-9
        valid = self.unions > 0  # classes that had any boundary pixels at all

        # exclude classes with no boundary (union==0) from macro average
        if self.average == "macro":
            denom = valid.sum()
            iou_per_class = self.intersections[valid] / (self.unions[valid] + eps)
            return iou_per_class.mean() if denom > 0 else torch.tensor(0.0)
        else:  # micro
            inter_sum = self.intersections[valid].sum()
            union_sum = self.unions[valid].sum() + eps
            return inter_sum / union_sum
