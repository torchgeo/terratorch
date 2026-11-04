import os

from vllm.model_executor.models.registry import ModelRegistry
from vllm.logger import init_logger

def _skip_token_validation_for_terratorch_renderer(logger):
    """Terratorch models consume raw multimodal data and are sent a dummy token id.

    Some vLLM versions (e.g. 0.33.0) validate prompt token ids against
    `model_config.get_vocab_size()` even when no tokenizer is initialized. Terratorch
    models have no vocabulary (vocab size is 0), so the dummy token is rejected with
    "Token id 1 is out of vocabulary". Token ids are meaningless for these models,
    so the validation is disabled on the Terratorch renderer.
    """
    try:
        from vllm.renderers.terratorch import TerratorchRenderer
    except ImportError:
        # Older vLLM versions have no dedicated Terratorch renderer
        return

    TerratorchRenderer.validate_token_ids = lambda self, token_ids, *, parameter=None: None
    logger.info("Disabled token id validation for the Terratorch renderer")

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

    _skip_token_validation_for_terratorch_renderer(logger)
