import logging

from terratorch.tasks.base_task import TerraTorchTask
from terratorch.tasks.classification_tasks import ClassificationTask
from terratorch.tasks.embedding_generation import EmbeddingGenerationTask
from terratorch.tasks.inference_task import InferenceTask
from terratorch.tasks.multilabel_classification_tasks import MultiLabelClassificationTask
from terratorch.tasks.embedding_decoding_task import EmbeddingClassificationTask, EmbeddingDecodingTask
from terratorch.tasks.object_detection_task import ObjectDetectionTask
from terratorch.tasks.reconstruction_tasks import ReconstructionTask
from terratorch.tasks.regression_tasks import PixelwiseRegressionTask, ScalarRegressionTask
from terratorch.tasks.segmentation_tasks import SemanticSegmentationTask

try:
    wxc_present = True
    from terratorch.tasks.wxc_downscaling_task import WxCDownscalingTask
    from terratorch.tasks.wxc_task import WxCTask

    logging.getLogger("terratorch").debug("wxc_downscaling found.")
except ImportError:
    import logging

    logging.getLogger("terratorch").debug("wxc_downscaling not installed")
    wxc_present = False


__all__ = (
    "ArSegmentationTask",
    "ClassificationTask",
    "EmbeddingGenerationTask",
    "InferenceTask",
    "MultiLabelClassificationTask",
    "EmbeddingClassificationTask",
    "EmbeddingDecodingTask",
    "ObjectDetectionTask",
    "PixelwiseRegressionTask",
    "ReconstructionTask",
    "ScalarRegressionTask",
    "SemanticSegmentationTask",
)

if wxc_present:
    __all__.__add__(
        (
            "WxCDownscalingTask",
            "WxCTask",
        )
    )
