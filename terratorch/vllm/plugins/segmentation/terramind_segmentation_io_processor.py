# Copyright contributors to the Terratorch project

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import copy

from collections.abc import Sequence
from typing import Any, Optional, Union, overload

import rasterio
import torch
from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorRequest, IOProcessorResponse
from vllm.inputs import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor, IOProcessorInput, IOProcessorOutput

from terratorch.tasks.tiled_inference import generate_tiled_inference_output, prepare_tiled_inference_input
from terratorch.vllm.plugins import generate_datamodule
from terratorch.cli_tools import write_tiff
from vllm.renderers import BaseRenderer

from .utils import download_file_sync, get_filename_from_url, path_or_tmpdir, to_base64_tiff

from .types import PluginConfig, RequestData, RequestOutput, TerramindSegmentationRequestInfo, TiledInferenceParameters

logger = logging.getLogger(__name__)


class TerramindSegmentationIOProcessor(IOProcessor):
    """vLLM IOProcessor for Terramind segmentation tasks

    This class instantiates an IO Processor plugin for vLLM for pre/post processing of multimodal data
    to be used with Terramind in Segmentation tasks.
    This plugin accepts multimodal data in the format of a url, or a directory path.
    Similarly, it can generate GeoTiff images is the form of a base64 encoded string or a file path.

    The plugin accepts and returns data in various formats and can be configured via the below environment variable:
        TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG
    This variable is to be set while starting the vLLM instance.
    The plugins configurable variables are:
    - output_path (String): Default path for storing output files when requesting output in 'path' mode. It is is ignored otherwise.
    The full schema of the plugin configuration can be found in vllm.plugins.segmentation.types.PluginConfig

    Once instantiated from the vLLM side, the plugin is automatically used when performing inference requests to the
    '/pooling' endpoint of a vLLM instance.
    """

    # The IO Processor plugin inerface requires the renderer argument after vLLM 0.16.0.
    # Support for vLLM <= v0.16.0 is deprecated and will be removed in TerraTorch v1.6.0.
    @overload
    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer) -> None: ...

    @overload
    def __init__(self, vllm_config: VllmConfig) -> None: ...

    def __init__(self, vllm_config: VllmConfig, renderer: Optional[BaseRenderer] = None):

        if renderer is None:
            logger.warning(
                "You are using a version of vLLM <= v0.16.0 that relies on the old IO Processor plugin interface. "
                "Support for vLLM <= v0.16.0 will be removed in TerraTorch v1.6.0."
            )
            super().__init__(vllm_config)
        else:
            super().__init__(vllm_config, renderer)

        self.model_config = vllm_config.model_config.hf_config.to_dict()["pretrained_cfg"]

        if not "data" in self.model_config:
            raise ValueError("The model config does not contain the Terratorch datamodule configuration")

        plugin_config_string = os.getenv("TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG", "{}")

        self.plugin_config = PluginConfig.model_validate_json(plugin_config_string)

        self.tiled_inference_parameters = self._init_tiled_inference_parameters_info()
        self.batch_size = 1
        self.requests_cache: dict[str, TerramindSegmentationRequestInfo] = {}

    def _init_tiled_inference_parameters_info(self) -> TiledInferenceParameters:
        if "tiled_inference_parameters" in self.model_config["model"]["init_args"]:
            tiled_inf_param_dict = self.model_config["model"]["init_args"]["tiled_inference_parameters"]
            if not all(["h_crop" in tiled_inf_param_dict, "w_crop" in tiled_inf_param_dict]):
                if "crop" in tiled_inf_param_dict:
                    tiled_inf_param_dict["h_crop"] = tiled_inf_param_dict["crop"]
                    tiled_inf_param_dict["w_crop"] = tiled_inf_param_dict["crop"]
                    del tiled_inf_param_dict["crop"]
                else:
                    raise ValueError(
                        f"Expect 'crop' (or 'h_crop' and 'w_crop') in tiled_inference_parameters "
                        f"but got {tiled_inf_param_dict}"
                    )
            if "stride" in tiled_inf_param_dict:
                tiled_inf_param_dict["h_stride"] = tiled_inf_param_dict["stride"]
                tiled_inf_param_dict["w_stride"] = tiled_inf_param_dict["stride"]
                del tiled_inf_param_dict["stride"]
        else:
            tiled_inf_param_dict = {}

        return TiledInferenceParameters(**tiled_inf_param_dict)

    def _get_datamodule_config(self) -> dict:
        data_module_config = copy.deepcopy(self.model_config["data"])

        if "ImpactMeshDataModule" in data_module_config["class_path"]:
            # This is so that we can put the input data in a folder with an arbitrary name.
            # However, this requires for the means and stds to be included in the model configuration
            data_module_config["init_args"]["label_grep"] = ""

        return data_module_config

    def _download_input_data(self, dataset_path: str, request_data: dict):
        # One thread per modality — downloads are I/O-bound so the GIL is
        # released during the blocking socket read, giving true parallelism.
        dir_path = Path(dataset_path)

        def _download_one(modality: str, url: str) -> None:
            modality_dir = dir_path / modality
            modality_dir.mkdir()
            dest_file = modality_dir / get_filename_from_url(url)
            download_file_sync(url, dest_file)

        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(_download_one, mod, url) for mod, url in request_data.items()]
            for f in as_completed(futures):
                f.result()  # re-raises any download exception

    def parse_request(self, request: Any) -> IOProcessorInput:
        logger.warning(
            "You are using a version of vLLM <= v0.16 that relies on the old IO Processor plugin interface. "
            "Support for vLLM <= v0.16 will be removed in TerraTorch v1.6.0."
        )
        return self.parse_data(request)

    def parse_data(self, data: Any) -> IOProcessorInput:
        if type(data) is dict:
            image_prompt = RequestData(**data)
            return image_prompt
        if isinstance(data, IOProcessorRequest):
            if not hasattr(data, "data"):
                raise ValueError("missing 'data' field in OpenAIBaseModel Request")

            request_data = data.data

            if type(request_data) is dict:
                return RequestData(**request_data)
            else:
                raise ValueError("Unable to parse the request data")

        raise ValueError("Unable to parse request")

    def output_to_response(self, plugin_output: IOProcessorOutput) -> IOProcessorResponse:
        return IOProcessorResponse(
            request_id=plugin_output.request_id,
            data=plugin_output,
        )

    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:

        request_data = RequestData.model_validate(prompt)

        input_data_format = request_data.data_format

        # Validate out_path if provided and out_data_format is "path"
        if request_data.out_data_format == "path" and request_data.out_path:
            if not os.path.exists(request_data.out_path):
                raise ValueError(f"The output path '{request_data.out_path}' does not exist")
            if not os.access(request_data.out_path, os.W_OK):
                raise ValueError(f"The output path '{request_data.out_path}' is not writable")

        datamodule_config = self._get_datamodule_config()

        with path_or_tmpdir(request_data) as dataset_path:
            if input_data_format == "url":
                self._download_input_data(dataset_path, request_data=request_data.data)

            # set the datamodule data_root to where the dataset is located
            datamodule_config["init_args"]["predict_data_root"] = dataset_path
            datamodule = generate_datamodule(datamodule_config)
            datamodule.batch_size = 1
            datamodule.setup("predict")

            # process the input files into Tensors
            data_loader = datamodule.predict_dataloader()
            data = list(data_loader)[0]

            # We can only retrieve metadata if the dataset contains a tiff file that we can load
            if "filename" in data:
                input_image_path = data["filename"][0]
                with rasterio.open(input_image_path, "r") as src:
                    metadata = src.meta
            else:
                logger.warning("No metadata can be loaded for the provided input")
                metadata = {}

        # Split the input in tiles depending on the tiled inference parameters
        input_data = datamodule.aug(data)["image"]
        prompt_data, tensor_reshape_fn, input_batch_size, h_img, w_img, _, delta = prepare_tiled_inference_input(
            input_data,
            **self.tiled_inference_parameters.model_dump(exclude={"average_patches"}),
        )

        prompts = []
        for tile in prompt_data:
            reshaped_tile = tensor_reshape_fn(tile.input_data)
            # TODO: Check if there's a better way of getting the data in the correct data type ouf of the box.
            multi_modal_data = {mod: tensor.to(torch.float16) for mod, tensor in reshaped_tile.items()}

            # Wrap in {"image": ...} as required by vLLM's multimodal input structure.
            # The version check was unreliable for dev installs; this path is always
            # correct for the vLLM versions this plugin supports (>= 0.14.0).
            multi_modal_data = {"image": multi_modal_data}

            prompt = {"prompt_token_ids": [1], "multi_modal_data": multi_modal_data}

            prompts.append(prompt)

        # if no request_id is passed this means that the plugin is used with vlLM
        # in offline sync mode. Therefore, we assume that one request at a time is being processed
        if not request_id:
            request_id = "offline"
        self.requests_cache[request_id] = TerramindSegmentationRequestInfo(
            out_data_format=request_data.out_data_format,
            out_path=request_data.out_path,
            dataset_path=dataset_path,
            prompt_data=prompt_data,
            h_img=h_img,
            w_img=w_img,
            input_batch_size=input_batch_size,
            metadata=metadata,
            filename=data["filename"][0],
            delta=delta,
        )

        return prompts

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Union[PromptType, Sequence[PromptType]]:
        return self.pre_process(prompt, request_id, **kwargs)

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> IOProcessorOutput:

        if not request_id:
            request_id = "offline"

        if request_id and (request_id in self.requests_cache):
            request_info = self.requests_cache[request_id]
            del self.requests_cache[request_id]

        output_format = request_info.out_data_format

        model_outputs = [output.outputs.data.squeeze(0) for output in model_output]
        outputs = list(zip(request_info.prompt_data, model_outputs, strict=True))
        output = generate_tiled_inference_output(
            outputs=outputs,
            input_batch_size=request_info.input_batch_size,
            h_img=request_info.h_img,
            w_img=request_info.w_img,
            delta=request_info.delta,
            padding=self.tiled_inference_parameters.padding,
            average_patches=self.tiled_inference_parameters.average_patches,
        )

        prediction = output.squeeze(0).argmax(dim=0).numpy()

        metadata = request_info.metadata

        ret: str
        if output_format == "path":
            # Use out_path from request if provided, otherwise use plugin config output_path
            output_dir = request_info.out_path if request_info.out_path else self.plugin_config.output_path
            out_file_path = Path(output_dir) / f"{Path(request_info.filename).stem}_prediction.tif"
            write_tiff(prediction, out_file_path, metadata)
            ret = str(out_file_path.resolve())
        elif output_format == "b64_json":
            ret = to_base64_tiff(prediction, metadata=metadata)

        return RequestOutput(
            data_format=request_info.out_data_format,
            data=ret,
            request_id=request_id,
        )
