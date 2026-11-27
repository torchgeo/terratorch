import logging

import huggingface_hub
import torch
from terratorch_surya.downstream_examples.ar_segmentation.models import (
    HelioSpectformer2D as HelioSpectformer2D_ar,
)
from terratorch_surya.downstream_examples.ar_segmentation.models import UNet as SuryaUNet
from terratorch_surya.downstream_examples.euv_spectra_prediction.models import (
    HelioSpectformer2D as HelioSpectformer2D_euv,
)
from terratorch_surya.downstream_examples.solar_flare_forecasting.models import (
    HelioSpectformer2D as HelioSpectformer2D_flare,
)
from terratorch_surya.downstream_examples.solar_wind_forecasting.models import (
    HelioSpectformer2D as HelioSpectformer2D_wind,
)
from terratorch_surya.models.helio_spectformer import HelioSpectFormer
from torch import nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


def load_weights(model: nn.Module, ckpt_data: dict, **kwargs) -> nn.Module:
    logging.getLogger("terratorch").info("Loading weights.")

    if ckpt_data is not None:
        download_kwargs = dict(ckpt_data)
        revision = download_kwargs.pop("revision", None)
        ckpt_data = huggingface_hub.hf_hub_download(
            **download_kwargs,
            revision=revision,  # nosec B615 - pinned in remote_checkpoint_path
        )

        checkpoint_model = torch.load(ckpt_data, map_location="cpu", weights_only=True)

        # load pre-trained model
        model.load_state_dict(checkpoint_model, strict=False)
        logging.getLogger("terratorch").info("Weights loaded.")

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def heliofm_backbone_surya(
    img_size: int = 4096,
    patch_size: int = 16,
    in_chans: int = 13,
    embed_dim: int = 1280,
    time_embedding: dict = {"type": "linear", "n_queries": None, "time_dim": 2},
    depth: int = 10,
    n_spectral_blocks: int = 2,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    window_size: int = 2,
    dp_rank: int = 8,
    learned_flow: bool = False,
    use_latitude_in_learned_flow: bool = False,
    init_weights: bool = False,
    checkpoint_layers: list = [],
    rpe: bool = False,
    ensemble: int = None,
    finetune: bool = False,
    ckpt_data: str = None,
    pretrained: bool = False,
    dtype="bfloat16",
):
    remote_checkpoint_path = {
        "repo_id": "nasa-ibm-ai4science/Surya-1.0",
        "filename": "surya.366m.v1.pt",
        "revision": "5cc4b5386d5f78fda3896b1389589d4e173bf212",
    }

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    model = HelioSpectFormer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        time_embedding=time_embedding,
        depth=depth,
        n_spectral_blocks=n_spectral_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        window_size=window_size,
        dp_rank=dp_rank,
        learned_flow=learned_flow,
        use_latitude_in_learned_flow=use_latitude_in_learned_flow,
        init_weights=init_weights,
        checkpoint_layers=checkpoint_layers,
        rpe=rpe,
        ensemble=ensemble,
        finetune=finetune,
        dtype=dtype,
    )

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def heliofm_backbone_surya_ar_segmentation(
    model_type: str = "spectformer_lora",
    img_size: int = 4096,
    patch_size: int = 16,
    in_chans: int = 13,
    embed_dim: int = 1280,
    time_embedding: dict = {"type": "linear", "n_queries": None, "time_dim": 1},
    depth: int = 10,
    n_spectral_blocks: int = 2,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    dtype: str = "bfloat16",
    window_size: int = 2,
    dp_rank: int = 4,
    learned_flow: bool = False,
    use_latitude_in_learned_flow: bool = False,
    init_weights: bool = False,
    checkpoint_layers: list = [],
    rpe: bool = False,
    finetune: bool = True,
    unet_embed_dim: int | None = None,
    unet_n_blocks: int | None = None,
    ckpt_data: str = None,
    pretrained: bool = False,
    config: dict = {
        "model": {
            "global_average_pooling": True,
            "global_max_pooling": False,
            "attention_pooling": False,
            "transformer_pooling": False,
            "dropout": False,
            "penultimate_linear_layer": False,
            "ft_unembedding_type": "linear",
            "ft_out_chans": 1,
        }
    },
):
    remote_checkpoint_path = {
        "repo_id": "nasa-ibm-ai4science/ar_segmentation_surya",
        "filename": "ar_segmentation_weights.pth",
    }

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    if model_type == "spectformer_lora":
        print("Initializing spectformer with LoRA.")
        model = HelioSpectformer2D_ar(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            dtype=dtype,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            checkpoint_layers=checkpoint_layers,
            rpe=rpe,
            finetune=finetune,
            config=config,
        )
    elif model_type == "unet":
        print("Initializing UNet.")
        model = SuryaUNet(
            in_chans=in_chans,
            embed_dim=unet_embed_dim,
            out_chans=1,
            n_blocks=unet_blocks,
        )
    else:
        raise ValueError(f"Unknown model type {model_type}.")

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def heliofm_backbone_surya_euv_spectra_prediction(
    model_type: str = "spectformer_lora",
    img_size: int = 4096,
    patch_size: int = 16,
    in_chans: int = 13,
    embed_dim: int = 1280,
    time_embedding: dict = {"type": "linear", "n_queries": None, "time_dim": 1},
    depth: int = 10,
    n_spectral_blocks: int = 2,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    dtype: str = "bfloat16",
    window_size: int = 2,
    dp_rank: int = 4,
    learned_flow: bool = False,
    use_latitude_in_learned_flow: bool = False,
    init_weights: bool = False,
    checkpoint_layers: list = [],
    rpe: bool = False,
    finetune: bool = True,
    unet_embed_dim: int | None = None,
    unet_n_blocks: int | None = None,
    ckpt_data: str = None,
    pretrained: bool = False,
    config: dict = {
        "model": {
            "global_average_pooling": False,
            "global_max_pooling": False,
            "attention_pooling": False,
            "transformer_pooling": False,
            "global_class_token": True,
            "dropout": 0.2,
            "penultimate_linear_layer": True,
            "ft_unembedding_type": "linear",
            "ft_out_chans": 1,
        }
    },
):
    remote_checkpoint_path = {
        "repo_id": "nasa-ibm-ai4science/euv_spectra_surya",
        "filename": "euv_spectra_weights.pth",
    }

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    if model_type == "spectformer_lora":
        print("Initializing spectformer with LoRA.")
        model = HelioSpectformer2D_euv(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            dtype=dtype,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            checkpoint_layers=checkpoint_layers,
            rpe=rpe,
            finetune=finetune,
            config=config,
        )
    elif model_type == "unet":
        print("Initializing UNet.")
        model = SuryaUNet(
            in_chans=in_chans,
            embed_dim=unet_embed_dim,
            out_chans=1,
            n_blocks=unet_blocks,
        )
    else:
        raise ValueError(f"Unknown model type {model_type}.")

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def heliofm_backbone_surya_solar_flare_forecasting(
    model_type: str = "spectformer_lora",
    img_size: int = 4096,
    patch_size: int = 16,
    in_chans: int = 13,
    embed_dim: int = 1280,
    time_embedding: dict = {"type": "linear", "n_queries": None, "time_dim": 1},
    depth: int = 10,
    n_spectral_blocks: int = 2,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    dtype: str = "bfloat16",
    window_size: int = 2,
    dp_rank: int = 4,
    learned_flow: bool = False,
    use_latitude_in_learned_flow: bool = False,
    init_weights: bool = False,
    checkpoint_layers: list = [],
    rpe: bool = False,
    finetune: bool = True,
    unet_embed_dim: int | None = None,
    unet_n_blocks: int | None = None,
    ckpt_data: str = None,
    pretrained: bool = False,
    config: dict = {
        "model": {
            "global_average_pooling": False,
            "global_max_pooling": False,
            "attention_pooling": False,
            "transformer_pooling": False,
            "global_class_token": True,
            "dropout": 0.2,
            "penultimate_linear_layer": True,
            "ft_unembedding_type": "linear",
            "ft_out_chans": 1,
        }
    },
):
    remote_checkpoint_path = {
        "repo_id": "nasa-ibm-ai4science/solar_flares_surya",
        "filename": "solar_flare_weights.pth",
    }

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    if model_type == "spectformer_lora":
        print("Initializing spectformer with LoRA.")
        model = HelioSpectformer2D_flare(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            dtype=dtype,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            checkpoint_layers=checkpoint_layers,
            rpe=rpe,
            finetune=finetune,
            config=config,
        )
    elif model_type == "unet":
        print("Initializing UNet.")
        model = SuryaUNet(
            in_chans=in_chans,
            embed_dim=unet_embed_dim,
            out_chans=1,
            n_blocks=unet_blocks,
        )
    else:
        raise ValueError(f"Unknown model type {model_type}.")

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def heliofm_backbone_surya_solar_wind_forecasting(
    model_type: str = "spectformer_lora",
    img_size: int = 4096,
    patch_size: int = 16,
    in_chans: int = 13,
    embed_dim: int = 1280,
    time_embedding: dict = {"type": "linear", "n_queries": None, "time_dim": 1},
    depth: int = 10,
    n_spectral_blocks: int = 2,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    dtype: str = "bfloat16",
    window_size: int = 2,
    dp_rank: int = 4,
    learned_flow: bool = False,
    use_latitude_in_learned_flow: bool = False,
    init_weights: bool = False,
    checkpoint_layers: list = [],
    rpe: bool = False,
    finetune: bool = True,
    unet_embed_dim: int | None = None,
    unet_n_blocks: int | None = None,
    ckpt_data: str = None,
    pretrained: bool = False,
    config: dict = {
        "model": {
            "global_average_pooling": False,
            "global_max_pooling": False,
            "attention_pooling": False,
            "transformer_pooling": False,
            "global_class_token": True,
            "dropout": 0.2,
            "penultimate_linear_layer": True,
            "ft_unembedding_type": "linear",
            "ft_out_chans": 1,
        }
    },
):
    remote_checkpoint_path = {
        "repo_id": "nasa-ibm-ai4science/solar_wind_surya",
        "filename": "solar_wind_weights.pth",
    }

    if not ckpt_data:
        ckpt_data = remote_checkpoint_path

    if model_type == "spectformer_lora":
        print("Initializing spectformer with LoRA.")
        model = HelioSpectformer2D_wind(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            dtype=dtype,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            checkpoint_layers=checkpoint_layers,
            rpe=rpe,
            finetune=finetune,
            config=config,
        )
    elif model_type == "unet":
        print("Initializing UNet.")
        model = SuryaUNet(
            in_chans=in_chans,
            embed_dim=unet_embed_dim,
            out_chans=1,
            n_blocks=unet_blocks,
        )
    else:
        raise ValueError(f"Unknown model type {model_type}.")

    if pretrained:
        model = load_weights(model, ckpt_data)

    return model
