import os

from vllm.logger import init_logger
from vllm.model_executor.models.registry import ModelRegistry


def register_terratorch_fix():
    logger = init_logger(__name__)

    # vLLM >= 0.31.0 defaults to Model Runner V2, whose pooling runner rejects
    # the "plugin" task that Terratorch models select, so every engine fails at
    # startup. vLLM has no per-model hook for choosing the runner, so default to
    # V1 here; this runs before the config is built. An explicit user setting
    # always wins.
    if "VLLM_USE_V2_MODEL_RUNNER" not in os.environ:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        logger.info(
            "Terratorch: defaulting VLLM_USE_V2_MODEL_RUNNER=0 because Model Runner V2 "
            "does not support the 'plugin' pooling task; set it explicitly to override"
        )

    try:
        ModelRegistry.register_model(
            "Terratorch",
            "terratorch.vllm.plugins.general.terratorch:TerratorchFix",
        )
        logger.info(
            "Successfully registered TerratorchFix model with vLLM, "
            "this is only required to support vLLM >0.21.0 <=0.24.0 due to a known bug"
        )
    except Exception as e:
        logger.error(f"Failed to register TerratorchFix model: {e}")
        raise
