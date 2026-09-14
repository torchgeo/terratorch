# Copyright contributors to the Terratorch project

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

from .utils import InputDefinition, InputTypeEnum


class TerratorchModelState(DefaultModelState):
    """Custom ModelState for the V2 runner.

    Terratorch passes raw mm tensors directly to forward() rather than going
    through the encoder-embedding path. This state extracts the tensors from
    the request features on real batches, and synthesises dummy tensors of the
    correct shape for warmup/profile passes.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        # input_definition describes each named tensor input (name → shape).
        pretrained_cfg = vllm_config.model_config.hf_config.to_dict()["pretrained_cfg"]
        self.input_definition = InputDefinition(**pretrained_cfg["input"])

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict:
        """Return raw mm tensors extracted from cached request features.

        Falls back to dummy tensors when requests carry no mm data (e.g.
        during the kernel-warmup pass, which builds NewRequestData with
        mm_features=[]).
        """
        # Collect (modality, MultiModalKwargsItem) pairs for requests in batch.
        mm_kwargs_list = []
        for req_id in input_batch.req_ids:
            if self.encoder_cache is None:
                break
            for feature in self.encoder_cache.mm_features.get(req_id, []):
                if feature.data is not None:
                    mm_kwargs_list.append((feature.modality, feature.data))

        if mm_kwargs_list:
            combined: dict = {}
            for _, _, batch in group_and_batch_mm_kwargs(
                mm_kwargs_list, device=self.device, pin_memory=PIN_MEMORY
            ):
                combined.update(batch)
            return combined

        # No real mm data (warmup / empty batch) — return dummy tensors so
        # forward() receives correctly-shaped inputs rather than None.
        return self.prepare_dummy_inputs(
            input_batch.num_reqs, input_batch.num_tokens
        )

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict:
        """Return dummy mm tensors of the correct shape for warmup runs.

        The real mm path goes through group_and_batch_mm_kwargs which calls
        MultiModalBatchedField._reduce_data and unsqueezes a batch dimension,
        producing (1, C, T, H, W) from a (C, T, H, W) InputDefinition shape.
        We replicate that here so the model receives the same shape.
        """
        dummy: dict = {}
        for name, inp in self.input_definition.data.items():
            if inp.type == InputTypeEnum.tensor:
                # unsqueeze(0) mirrors what _reduce_data does for a single item
                dummy[name] = torch.full(
                    [1, *inp.shape], 1.0, dtype=torch.float16, device=self.device
                )
        return dummy

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states):
        # Terratorch does not use inputs_embeds; raw mm kwargs are passed
        # directly to forward() via prepare_inputs().
        return None

    def dummy_inputs_embeds(self, num_tokens: int):
        return None
