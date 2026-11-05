import importlib
import logging
import os
import typing

import torch
from torch import nn

from terratorch.models.model import (
    Model,
    ModelFactory,
    ModelOutput,
)
from terratorch.models.pincers.unet_pincer import UNetPincer
from terratorch.models.pincers.wxc_downscaling_pincer import get_downscaling_pincer
from terratorch.registry import MODEL_FACTORY_REGISTRY

logger = logging.getLogger(__name__)


class WxCModuleWrapper(Model, nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def freeze_encoder(self):
        logger.info("freeze encoder")
        for param in self.module.backbone.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        logger.info("freeze decoder")
        for param in self.module.head.parameters():
            param.requires_grad = False

    def forward(self, x) -> ModelOutput:
        mo = self.module.forward(x)
        return ModelOutput(mo)

    def load_state_dict(self, state_dict: os.Mapping[str, typing.Any], strict: bool = True, assign: bool = False):
        self.module.load_state_dict(state_dict, strict, assign)


@MODEL_FACTORY_REGISTRY.register
class WxCModelFactory(ModelFactory):
    def build_model(
        self,
        backbone: str | nn.Module,
        aux_decoders: str,
        checkpoint_path: str = None,
        backbone_weights: str = None,
        **kwargs,
    ) -> Model:
        if backbone == "prithviwxc":
            try:
                prithviwxc = importlib.import_module("PrithviWxC.model")
            except ModuleNotFoundError as e:
                print(f"Module not found: {e.name}. Please install PrithviWxC using pip install PrithviWxC")
                raise

            # remove parameters not meant for the backbone but for other parts of the model
            logger.debug(kwargs)
            if "skip_connection" in kwargs.keys():
                skip_connection = kwargs.pop("skip_connection")
            else:
                skip_connection = True
            if "wxc_auxiliary_data_path" in kwargs.keys():
                wxc_auxiliary_data_path = kwargs.pop("wxc_auxiliary_data_path")
            if "config_path" in kwargs.keys():
                config_path = kwargs.pop("config_path")
                # from granitewxc.utils.config import get_config #TODO rkie fix: import flaky
                import yaml
                from granitewxc.utils.config import ExperimentConfig

                def get_config(config_path: str) -> ExperimentConfig:
                    cfg = yaml.safe_load(open(config_path))
                    return ExperimentConfig.from_dict(cfg)

                # end TODO
                config = get_config(config_path)

            if aux_decoders == "unetpincer":
                backbone = prithviwxc.PrithviWxC(**kwargs)
                # Freeze PrithviWxC model parameters
                for param in backbone.parameters():
                    param.requires_grad = False

                # Load pre-trained weights if checkpoint is provided
                if backbone_weights is not None:
                    print(f"Starting to load model from {backbone_weights}")
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    state_dict = torch.load(
                        f=backbone_weights,
                        weights_only=True,
                        map_location=torch.device(device),
                    )

                    # Compare the keys in model and saved state_dict
                    model_keys = set(backbone.state_dict().keys())
                    saved_state_dict_keys = set(state_dict.keys())

                    # Find keys that are in the model but not in the saved state_dict
                    missing_in_saved = model_keys - saved_state_dict_keys
                    # Find keys that are in the saved state_dict but not in the model
                    missing_in_model = saved_state_dict_keys - model_keys
                    # Find keys that are common between the model and the saved state_dict
                    common_keys = model_keys & saved_state_dict_keys

                    # Print the common keys
                    if common_keys:
                        print(f"Keys loaded : {common_keys}")

                    # Print the discrepancies
                    if missing_in_saved:
                        print(f"Keys present in model but missing in saved state_dict: {missing_in_saved}")
                    if missing_in_model:
                        print(f"Keys present in saved state_dict but missing in model: {missing_in_model}")

                    # Load the state_dict with strict=False to allow partial loading
                    backbone.load_state_dict(state_dict=state_dict, strict=False)
                    print("=>" * 10, f"Model loaded from {backbone_weights}...")
                    print("Loaded backbone weights")
                else:
                    print("Not loading backbone model weigts")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                backbone.to(device)
                model_to_return = UNetPincer(backbone, skip_connection=skip_connection).to(device)
                return model_to_return
            if aux_decoders == "downscaler":
                try:
                    __import__("granitewxc.utils.config")
                    from granitewxc.utils.config import get_config
                    from granitewxc.utils.downscaling_model import get_backbone, get_finetune_model

                    config.data.data_path_surface = os.path.join(wxc_auxiliary_data_path, "merra-2")
                    config.data.data_path_vertical = os.path.join(wxc_auxiliary_data_path, "merra-2")
                    config.data.climatology_path_surface = os.path.join(wxc_auxiliary_data_path, "climatology")
                    config.data.climatology_path_vertical = os.path.join(wxc_auxiliary_data_path, "climatology")
                    config.model.input_scalers_surface_path = os.path.join(
                        wxc_auxiliary_data_path, "climatology/musigma_surface.nc"
                    )
                    config.model.input_scalers_vertical_path = os.path.join(
                        wxc_auxiliary_data_path, "climatology/musigma_vertical.nc"
                    )
                    config.model.output_scalers_surface_path = os.path.join(
                        wxc_auxiliary_data_path, "climatology/anomaly_variance_surface.nc"
                    )
                    config.model.output_scalers_vertical_path = os.path.join(
                        wxc_auxiliary_data_path, "climatology/anomaly_variance_vertical.nc"
                    )
                    backbone = get_backbone(config)
                except ImportError as e:
                    print(f"Module not found: {e.name}. Please install granitewxc using pip install granitewxc")
                dsp = get_downscaling_pincer(config, backbone)
                if checkpoint_path:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dsp.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
                return dsp
            return WxCModuleWrapper(backbone)

        elif backbone == "prithvi-eccc-downscaling":
            try:
                model_args = kwargs["model_args"]
                if model_args.model.unet:
                    from granitewxc.models.model import get_finetune_model_UNET

                    module = get_finetune_model_UNET(model_args)
                else:
                    from granitewxc.models.model import get_finetune_model

                    module = get_finetune_model(model_args)

                if checkpoint_path:
                    weights_path = model_args.path_model_weights
                    weights = torch.load(weights_path, weights_only=True)["model"]
                    try:
                        module.load_state_dict(weights)
                    except Exception:
                        # remove the module prefix from the weights (for models trained with DDP)
                        weights = {key.replace("model.module.", ""): value for key, value in weights.items()}
                        module.load_state_dict(weights)

                if model_args.path_backbone_weights:
                    module.load_pretrained_backbone(
                        weights_path=model_args.path_backbone_weights,
                        sel_prefix=model_args.backbone_prefix,
                        freeze=model_args.backbone_freeze,
                    )

                return WxCModuleWrapper(module)
            except ImportError:
                print("ECCC downscaling module not found")
                print("To install it, run:")
                print("pip install git+https://github.com/IBM/granite-wxc.git")

        # starting from there only for backwards compatibility, deprecated
        if backbone == "gravitywave":
            try:
                __import__("prithviwxc.gravitywave.inference")
                from prithviwxc.gravitywave.config import get_cfg
                from prithviwxc.gravitywave.inference import get_model

                cfg = get_cfg()
                model_wrapper = WxCModuleWrapper(get_model(cfg, "uvtp122", cfg.singular_sharded_checkpoint))
                if checkpoint_path:
                    model_wrapper.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                return model_wrapper
            except ImportError as e:
                missing_module = e.name if hasattr(e, "name") else "unknown module"
                print("prithvi wxc gravitywave not installed, missing module: {missing_module}")
                return None
        else:
            try:
                __import__("granitewxc.utils.config")
                from granitewxc.utils.config import get_config
                from granitewxc.utils.downscaling_model import get_finetune_model

                module = get_finetune_model(kwargs["model_config"])
                model_wrapper = WxCModuleWrapper(module)

                if checkpoint_path:
                    model_wrapper.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                return model_wrapper
            except ImportError:
                print("granite wxc downscaling not installed")
