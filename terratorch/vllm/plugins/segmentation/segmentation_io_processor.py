# Copyright contributors to the Terratorch project

from __future__ import annotations

import base64
import copy
import logging
import os
import tempfile
import urllib.request
import uuid
import warnings
from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, overload

import numpy as np
import rasterio
import regex as re
import torch
from einops import rearrange
from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorRequest, IOProcessorResponse
from vllm.inputs import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor, IOProcessorInput, IOProcessorOutput

from terratorch.vllm.plugins import generate_datamodule
from vllm.renderers import BaseRenderer

from .types import PluginConfig, RequestData, RequestOutput, SegmentationRequestInfo, TiledInferenceParameters

logger = logging.getLogger(__name__)

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
OFFSET = 0
PERCENTILE = 99

DEFAULT_INPUT_INDICES = [0, 1, 2, 3, 4, 5]


class SegmentationIOProcessor(IOProcessor):
    """vLLM IOProcessor for segmentation tasks

    This class instantiates an IO Processor plugin for vLLM for pre/post processing of GeoTiff images
    to be used with Segmentation tasks.
    This plugin accepts GeoTiff images in the format of a url, a base64 encoded string or a file path.
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

        if "data" not in self.model_config:
            raise ValueError("The model config does not contain the Terratorch datamodule configuration")

        plugin_config_string = os.getenv("TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG", "{}")

        self.plugin_config = PluginConfig.model_validate_json(plugin_config_string)

        self.datamodule = generate_datamodule(self.model_config["data"])
        
        # Store the augmentation template for creating per-request instances
        # This avoids shared mutable state and eliminates race conditions
        self._aug_template = self.datamodule.aug
        
        # Fix data_keys in the template if needed
        if hasattr(self._aug_template, 'transform_op') and hasattr(self._aug_template.transform_op, 'data_keys'):
            if self._aug_template.transform_op.data_keys is None:
                self._aug_template.transform_op.data_keys = ["image"]

        self.tiled_inference_parameters = self._init_tiled_inference_parameters_info()
        self.batch_size = 1
        self.requests_cache: dict[str, SegmentationRequestInfo] = {}

    def _init_tiled_inference_parameters_info(self) -> TiledInferenceParameters:
        if "tiled_inference_parameters" in self.model_config["model"]["init_args"]:
            tiled_inf_param_dict = self.model_config["model"]["init_args"]["tiled_inference_parameters"]
            if not all(["h_crop" in tiled_inf_param_dict, "w_crop" in tiled_inf_param_dict]):
                if "crop" in tiled_inf_param_dict:
                    tiled_inf_param_dict["h_crop"] = tiled_inf_param_dict["crop"]
                    tiled_inf_param_dict["w_crop"] = tiled_inf_param_dict["crop"]
                else:
                    raise ValueError(
                        f"Expect 'crop' (or 'h_crop' and 'w_crop') in tiled_inference_parameters "
                        f"but got {tiled_inf_param_dict}"
                    )
            if (
                "stride" in tiled_inf_param_dict
                or "w_stride" in tiled_inf_param_dict
                or "h_stride" in tiled_inf_param_dict
            ):
                warnings.warn("The 'stride' parameters for tiled inference are ignored in vLLM.")
        else:
            tiled_inf_param_dict = {}

        return TiledInferenceParameters(**tiled_inf_param_dict)

    def save_geotiff(
        self,
        image: torch.Tensor,
        meta: dict,
        out_format: str,
        request_id: str | None = None,
        output_path: str | None = None,
    ) -> str | bytes:
        """Save multi-band image in Geotiff file.

        Args:
            image: np.ndarray with shape (bands, height, width)
            meta: dict with meta info.
            out_format: output format ('path' or 'b64_json')
            request_id: request identifier for filename
            output_path: path where to save the image (used when out_format is 'path')
        """
        if out_format == "path":
            # Use provided output_path or fall back to plugin config
            output_dir = output_path if output_path else self.plugin_config.output_path
            if request_id:
                fname = f"{request_id}.tiff"
            else:
                fname = f"{uuid.uuid4()!s}.tiff"
            file_path = Path(output_dir) / fname
            with rasterio.open(str(file_path), "w", **meta) as dest:
                for i in range(image.shape[0]):
                    dest.write(image[i, :, :], i + 1)

            return str(file_path)
        elif out_format == "b64_json":
            with tempfile.NamedTemporaryFile() as tmpfile:
                with rasterio.open(tmpfile.name, "w", **meta) as dest:
                    for i in range(image.shape[0]):
                        dest.write(image[i, :, :], i + 1)

                file_data = tmpfile.read()
                return base64.b64encode(file_data).decode("utf-8")

        else:
            raise ValueError("Unknown output format")

    def _convert_np_uint8(self, float_image: torch.Tensor):
        image = float_image.numpy() * 255.0
        image = image.astype(dtype=np.uint8)

        return image

    def read_geotiff(
        self,
        file_path: str,
        path_type: str,
    ) -> tuple[np.ndarray, dict, tuple[float, float]]:
        """Read all bands from *file_path* and return image + meta info.

        Args:
            file_path: path to image file.

        Returns:
            np.ndarray with shape (bands, height, width)
            meta info dict
        """
        if all([x is None for x in [file_path, path_type]]):
            raise Exception("All input fields to read_geotiff are None")

        data: BytesIO
        if file_path is not None and path_type == "url":
            with urllib.request.urlopen(file_path) as response:
                data = BytesIO(response.read())
        elif file_path is not None and path_type == "path":
            with open(file_path, "rb") as f:
                data = BytesIO(f.read())
        elif file_path is not None and path_type == "b64_json":
            image_data = base64.b64decode(file_path)
            data = BytesIO(image_data)
        else:
            raise Exception("Wrong combination of parameters to read_geotiff")

        with rasterio.open(data) as src:
            img = src.read()
            meta = src.meta
            try:
                coords = src.lnglat()
            except:
                # Cannot read coords
                coords = None
        return img, meta, coords

    def load_image(
        self,
        data: list[str],
        path_type: str,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        indices: list[int] | None | None = None,
    ):
        """Build an input example by loading images in *file_paths*.

        Args:
            file_paths: list of file paths .
            mean: list containing mean values for each band in the
                images in *file_paths*.
            std: list containing std values for each band in the
                images in *file_paths*.

        Returns:
            np.array containing created example
            list of meta info for each image in *file_paths*
        """

        imgs = []
        metas = []
        temporal_coords = []
        location_coords = []

        for file in data:
            img, meta, coords = self.read_geotiff(file_path=file, path_type=path_type)
            # Rescaling (don't normalize on nodata)
            img = np.moveaxis(img, 0, -1)  # channels last for rescaling
            if indices is not None:
                img = img[..., indices]
            if mean is not None and std is not None:
                img = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - mean) / std)

            imgs.append(img)
            metas.append(meta)
            if coords is not None:
                location_coords.append(coords)

            try:
                match = re.search(r"(\d{7,8}T\d{6})", file)
                if match:
                    year = int(match.group(1)[:4])
                    julian_day = match.group(1).split("T")[0][4:]
                    if len(julian_day) == 3:
                        julian_day = int(julian_day)
                    else:
                        julian_day = datetime.strptime(julian_day, "%m%d").timetuple().tm_yday
                    temporal_coords.append([year, julian_day])
            except Exception:
                logger.exception("Could not extract timestamp for %s", file)

        imgs = np.stack(imgs, axis=0)  # num_frames, H, W, C
        imgs = np.moveaxis(imgs, -1, 0).astype("float32")  # C, num_frames, H, W
        imgs = np.expand_dims(imgs, axis=0)  # add batch di

        return imgs, temporal_coords, location_coords, metas

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
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:

        preprocess_start = datetime.now()
        image_data = dict(prompt)

        # Validate out_path if provided and out_data_format is "path"
        if image_data.get("out_data_format") == "path" and image_data.get("out_path"):
            out_path = Path(image_data["out_path"])
            if not out_path.exists():
                raise ValueError(f"The output path '{image_data['out_path']}' does not exist")
            if not os.access(str(out_path), os.W_OK):
                raise ValueError(f"The output path '{image_data['out_path']}' is not writable")

        indices = DEFAULT_INPUT_INDICES if not image_data["indices"] else image_data["indices"]

        input_data, temporal_coords, location_coords, meta_data = self.load_image(
            data=[image_data["data"]],
            indices=indices,
            path_type=image_data["data_format"],
        )

        if input_data.mean() > 1:
            input_data = input_data / 10000  # Convert to range 0-1

        original_h, original_w = input_data.shape[-2:]
        pad_h = (
            self.tiled_inference_parameters.h_crop - (original_h % self.tiled_inference_parameters.h_crop)
        ) % self.tiled_inference_parameters.h_crop
        pad_w = (
            self.tiled_inference_parameters.w_crop - (original_w % self.tiled_inference_parameters.w_crop)
        ) % self.tiled_inference_parameters.w_crop
        input_data = np.pad(
            input_data,
            ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="reflect",
        )

        batch = torch.tensor(input_data)
        windows = batch.unfold(
            3, self.tiled_inference_parameters.h_crop, self.tiled_inference_parameters.w_crop
        ).unfold(4, self.tiled_inference_parameters.h_crop, self.tiled_inference_parameters.w_crop)

        h1, w1 = windows.shape[3:5]
        windows = rearrange(
            windows,
            "b c t h1 w1 h w -> (b h1 w1) c t h w",
            h=self.tiled_inference_parameters.h_crop,
            w=self.tiled_inference_parameters.w_crop,
        )

        # if no request_id is passed this means that the plugin is used with vlLM
        # in offline sync mode. Therefore, we assume that one request at a time is being processed
        if not request_id:
            request_id = "offline"
        self.requests_cache[request_id] = SegmentationRequestInfo(
            out_data_format=image_data["out_data_format"],
            out_path=image_data.get("out_path"),
            metadata=meta_data[0],
            original_h=original_h,
            original_w=original_w,
            h1=h1,
            w1=w1,
        )

        # Split into batches if number of windows > batch_size
        num_batches = windows.shape[0] // self.batch_size if windows.shape[0] > self.batch_size else 1
        windows = torch.tensor_split(windows, num_batches, dim=0)

        if temporal_coords:
            temporal_coords = torch.tensor(temporal_coords).unsqueeze(0)
        else:
            temporal_coords = None
        if location_coords:
            location_coords = torch.tensor(location_coords[0]).unsqueeze(0).to(torch.float16)
        else:
            location_coords = None

        prompts = []
        # Create a per-request augmentation instance to avoid shared mutable state
        # This eliminates race conditions without needing locks
        aug = copy.deepcopy(self._aug_template)
        
        # Ensure data_keys is properly set after deepcopy (deepcopy may not preserve it correctly)
        if hasattr(aug, 'transform_op') and hasattr(aug.transform_op, 'data_keys'):
            if aug.transform_op.data_keys is None:
                aug.transform_op.data_keys = ["image"]
        
        for window in windows:
            # Apply standardization
            window = self.datamodule.test_transform(image=window.squeeze().numpy().transpose(1, 2, 0))
            
            try:
                window = aug(window)["image"]
            except Exception as e:
                logger.warning(f"First augmentation attempt failed: {e}. Retrying with adjusted dimensions.")
                window["image"] = window["image"][None, :, :, :]
                window = aug(window)["image"]

            multi_modal_data = {
                "pixel_values": window.to(torch.float16)[0],
            }
            # not all models use location coordinates, so we don't bother sending them to vLLM if not needed
            if "location_coords" in self.model_config["input"]["data"] and location_coords is not None:
                multi_modal_data["location_coords"] = location_coords

            prompt = {"prompt_token_ids": [1], "multi_modal_data": {"image": multi_modal_data}}

            prompts.append(prompt)

        return prompts

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:

        pred_imgs_list = []

        if not request_id:
            request_id = "offline"

        if request_id and (request_id in self.requests_cache):
            request_info = self.requests_cache[request_id]
            del self.requests_cache[request_id]

        for output in model_output:
            output_data = output.outputs.data
            if output_data.ndim == 3:
                argmax_dim = 0
                extend_dims = True
            elif output_data.ndim == 4:
                argmax_dim = 1
                extend_dims = False
            else:
                raise ValueError(
                    "The post-process function of the Terratorch Segmentation plugin "
                    f"got a tensor with {output_data.ndim} dimensions while it expects a 3 or 4 dimensional tensor."
                )
            y_hat = output_data.argmax(dim=argmax_dim).unsqueeze(0)
            if extend_dims:
                y_hat = y_hat.unsqueeze(0)
            pred = torch.nn.functional.interpolate(
                y_hat.float(),
                size=self.tiled_inference_parameters.h_crop,
                mode="nearest",
            )
            pred_imgs_list.append(pred)

        pred_imgs: torch.Tensor = torch.concat(pred_imgs_list, dim=0)

        # Build images from patches
        pred_imgs = rearrange(
            pred_imgs,
            "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
            h=self.tiled_inference_parameters.h_crop,
            w=self.tiled_inference_parameters.w_crop,
            b=1,
            c=1,
            h1=request_info.h1,
            w1=request_info.w1,
        )

        # Cut padded area back to original size
        pred_imgs = pred_imgs[..., : request_info.original_h, : request_info.original_w]

        # Squeeze (batch size 1)
        pred_imgs = pred_imgs[0]

        metadata = request_info.metadata
        metadata.update(count=1, dtype="uint8", compress="lzw", nodata=0)

        # Use out_path from request if provided, otherwise use plugin config output_path
        output_path = request_info.out_path if request_info.out_path else self.plugin_config.output_path
        out_data = self.save_geotiff(
            self._convert_np_uint8(pred_imgs), metadata, request_info.out_data_format, request_id, output_path
        )

        return RequestOutput(data_format=request_info.out_data_format, data=out_data, request_id=request_id)
