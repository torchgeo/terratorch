# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings

import torch
from huggingface_hub import hf_hub_download

from terratorch.registry import TERRATORCH_FULL_MODEL_REGISTRY

logger = logging.getLogger("terramind")

try:
    from .vqvae import VQVAE, DiVAE

    vqvae_available = True
    import_error = None
except Exception as e:
    logger.debug(f"Could not import TerraMind tokenizer due to ImportError({e})")
    vqvae_available = False
    import_error = e

try:
    from .text.text_tokenizer import CaptionTokenizer, CoordsTokenizer

    tokenizers_available = True
    import_error_tokenizers = None
except Exception as e:
    logger.debug(f"Could not import tokenizers due to ImportError({e})")
    tokenizers_available = False
    import_error_tokenizers = e


# Model definitions
__all__ = [
    "terramind_v01_caption_tokenizer",
    "terramind_v01_tokenizer_dem",
    "terramind_v01_tokenizer_lulc",
    "terramind_v01_tokenizer_s1grd",
    "terramind_v01_tokenizer_s2l2a",
    "terramind_v1_coords_tokenizer",
    "terramind_v1_tokenizer_dem",
    "terramind_v1_tokenizer_lulc",
    "terramind_v1_tokenizer_ndvi",
    "terramind_v1_tokenizer_s1grd",
    "terramind_v1_tokenizer_s1rtc",
    "terramind_v1_tokenizer_s2l2a",
]

pretrained_weights = {
    "terramind_v1_tokenizer_s2l2a": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S2L2A",
        "hf_hub_filename": "TerraMind_Tokenizer_S2L2A.pt",
    },
    "terramind_v1_tokenizer_s1rtc": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1RTC",
        "hf_hub_filename": "TerraMind_Tokenizer_S1RTC.pt",
    },
    "terramind_v1_tokenizer_s1grd": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1GRD",
        "hf_hub_filename": "TerraMind_Tokenizer_S1GRD.pt",
    },
    "terramind_v1_tokenizer_dem": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM",
        "hf_hub_filename": "TerraMind_Tokenizer_DEM.pt",
    },
    "terramind_v1_tokenizer_lulc": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-LULC",
        "hf_hub_filename": "TerraMind_Tokenizer_LULC.pt",
    },
    "terramind_v1_tokenizer_ndvi": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-NDVI",
        "hf_hub_filename": "TerraMind_Tokenizer_NDVI.pt",
    },
    "terramind_v1_coords_tokenizer": {
        "hf_hub_id": "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-Coords",
        "hf_hub_filename": "config.json",
    },
    "terramind_v01_tokenizer_s2l2a": {
        "hf_hub_id": "FAST-EO/TerraMind-v0.1-Tokenizer-S2L2A",
        "hf_hub_filename": "TerraMind_v01_Tokenizer_S2L2A.pt",
    },
    "terramind_v01_tokenizer_s1grd": {
        "hf_hub_id": "FAST-EO/TerraMind-v0.1-Tokenizer-S1GRD",
        "hf_hub_filename": "TerraMind_v01_Tokenizer_S1GRD.pt",
    },
    "terramind_v01_tokenizer_dem": {
        "hf_hub_id": "FAST-EO/TerraMind-v0.1-Tokenizer-DEM",
        "hf_hub_filename": "TerraMind_v01_Tokenizer_DEM.pt",
    },
    "terramind_v01_tokenizer_lulc": {
        "hf_hub_id": "FAST-EO/TerraMind-v0.1-Tokenizer-LULC",
        "hf_hub_filename": "TerraMind_v01_Tokenizer_LULC.pt",
    },
    "terramind_v01_caption_tokenizer": {
        "hf_hub_id": "FAST-EO/TerraMind-v0.1-Tokenizer-Text",
        "hf_hub_filename": "config.json",
    },
}


def build_vqvae(
    model_type: str = "divae", variant: str = None, pretrained: bool = False, ckpt_path: str | None = None, **kwargs
):

    if not vqvae_available:
        warnings.warn(
            "Cannot import VQVAE/DiVAE from terramind. \nMake sure to install `pip install diffusers==0.30.0`."
        )
        raise import_error

    if model_type == "divae":
        model = DiVAE(**kwargs)
    elif model_type == "vqvae":
        model = VQVAE(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if ckpt_path is not None:
        # Load model from checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        loaded_keys = model.load_state_dict(state_dict, strict=False)
        if loaded_keys.missing_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            logger.warning(f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}")

    elif pretrained:
        # Load model from Hugging Face
        state_dict_file = hf_hub_download(
            repo_id=pretrained_weights[variant]["hf_hub_id"], filename=pretrained_weights[variant]["hf_hub_filename"]
        )
        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)

    return model


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_tokenizer_s2l2a(**kwargs):
    """
    S2L2A Tokenizer for TerraMind v1.

    model args: Namespace(patch_size=16, input_size_min=224, input_size_max=256, resolution_step=32, input_size_enc=None, encoder_type="vit_b_enc", decoder_type="unet_patched", post_mlp=True, encoder_ckpt=None, full_ckpt=None, freeze_enc=False, from_scratch=True, dec_transformer_dropout=0.2, quantizer_type="fsq", codebook_size="8-8-8-6-5", latent_dim=5, norm_codes=True, norm_latents=False, codebook_weight=1.0, quantizer_ema_decay=0.99, coef_ema_dead_code=32.0, code_replacement_policy="batch_random", commitment_weight=1.0, kmeans_init=False, num_train_timesteps=1000, prediction_type="sample", beta_schedule="linear", zero_terminal_snr=True, cls_free_guidance_dropout=0.0, masked_cfg=False, masked_cfg_low=0, masked_cfg_high=None, thresholding=True, loss_fn="mse", conditioning="concat", resolution_cond=False, eval_res_cond=None, batch_size=1, accum_grad=1, batch_size_eval=None, epochs=100, stop_after_epoch=250, save_ckpt_freq=1, opt="adamw", opt_eps=1e-08, opt_betas=[0.9, 0.99], clip_grad=1.0, skip_grad=None, momentum=0.9, weight_decay=0.05, weight_decay_end=0.05, blr=0.0064, warmup_lr=1e-06, min_lr=0.0, warmup_epochs=1, warmup_steps=-1, dtype="fp16", model_ema=True, model_ema_decay=0.9999, model_ema_force_cpu=False, model_ema_update_freq=1, hflip=0.5, domain="sen2l2a@264", mask_value=None, data_path=["./data/MajorTOM-Core/train", "./data/SSL4EOS12/train_single"], eval_data_path="./data/MajorTOM-Core/val", imagenet_default_mean_and_std=False, standardize_surface_normals=False, min_crop_scale=0.8, cache_datasets=False, dataset_size=None, sample_sentinel_data=True, num_seasons=1, num_locations=64, dist_eval=True, step_eval=True, eval_noise_schedule="DDIMScheduler", num_eval_timesteps=50, input_size_eval=[256], num_eval_metrics_samples=50000, eval_freq=2000, eval_metrics_freq=0, eval_image_log_freq=2000, num_logged_images=15, eval_only=False, no_inception=False, log_codebook_usage=True, output_dir="output/tokenization/divae/rgb/Sentinel2-ViTB-UNetP4_16k_224-264/2k_continue", device="cuda", seed=0, resume="output/tokenization/divae/rgb/Sentinel2-ViTB-UNetP4_16k_224-264/2k_continue/checkpoint_29.pth", auto_resume=True, start_epoch=30, num_workers=5, pin_mem=True, find_unused_params=True, log_wandb=True, wandb_project=None, wandb_entity="erik-scheurer", wandb_run_name="2048_S2TerraMesh_divae/rgb/Sentinel2-ViTB-UNetP4_16k_224-264", wandb_tags=[], show_user_warnings=False, world_size=32, local_rank=-1, dist_on_itp=False, dist_url="env://", run_name="tokenization/divae/rgb/Sentinel2-ViTB-UNetP4_16k_224-264", patch_size_dec=4, clip_sample=True, epoch_eval=False, config_path="/p/project/geofm4eo/scheurer2/4m4eo/cfgs/default/tokenization/divae/rgb/Sentinel2-ViTB-UNetP4_16k_224-264.yaml", rank=0, gpu=0, distributed=True, dist_backend="nccl", num_tasks=32, all_domains=["sen2l2a@264"], input_size=256, lr=0.0001414213562373095, effective_batch_size=2048)
    """
    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_s2l2a",
        image_size=256,
        n_channels=12,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=kwargs.pop("clip_sample", True),
        **kwargs,
    )

    return tokenizer


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_tokenizer_s1rtc(**kwargs):
    """
    S1RTC Tokenizer for TerraMind v1.

    model args: Namespace(patch_size=16, input_size_min=224, input_size_max=256, resolution_step=32, input_size_enc=None, encoder_type="vit_b_enc", decoder_type="unet_patched", post_mlp=True, encoder_ckpt=None, full_ckpt=None, freeze_enc=False, from_scratch=True, dec_transformer_dropout=0.2, quantizer_type="fsq", codebook_size="8-8-8-6-5", latent_dim=5, norm_codes=True, norm_latents=False, codebook_weight=1.0, quantizer_ema_decay=0.99, coef_ema_dead_code=32.0, code_replacement_policy="batch_random", commitment_weight=1.0, kmeans_init=False, num_train_timesteps=1000, prediction_type="sample", beta_schedule="linear", zero_terminal_snr=True, cls_free_guidance_dropout=0.0, masked_cfg=False, masked_cfg_low=0, masked_cfg_high=None, thresholding=True, loss_fn="mse", conditioning="concat", resolution_cond=False, eval_res_cond=None, batch_size=1, accum_grad=1, batch_size_eval=None, epochs=100, stop_after_epoch=250, save_ckpt_freq=1, opt="adamw", opt_eps=1e-08, opt_betas=[0.9, 0.99], clip_grad=1.0, skip_grad=None, momentum=0.9, weight_decay=0.05, weight_decay_end=0.05, blr=0.0064, warmup_lr=1e-06, min_lr=0.0, warmup_epochs=1, warmup_steps=-1, dtype="fp16", model_ema=True, model_ema_decay=0.9999, model_ema_force_cpu=False, model_ema_update_freq=1, hflip=0.5, domain="sen1@264", mask_value=None, data_path="./data/MajorTOM-Core/train/", eval_data_path="./data/MajorTOM-Core/val", imagenet_default_mean_and_std=False, standardize_surface_normals=False, min_crop_scale=0.8, cache_datasets=False, dataset_size=None, sample_sentinel_data=True, num_seasons=1, num_locations=64, dist_eval=True, step_eval=True, eval_noise_schedule="DDIMScheduler", num_eval_timesteps=50, input_size_eval=[256], num_eval_metrics_samples=50000, eval_freq=1000, eval_metrics_freq=0, eval_image_log_freq=1000, num_logged_images=5, eval_only=False, no_inception=False, log_codebook_usage=True, output_dir="output/tokenization/divae/rgb/Sentinel1-ViTB-UNetP4_16k_224-264/2k_MajorTOM224256", device="cuda", seed=0, resume="output/tokenization/divae/rgb/Sentinel1-ViTB-UNetP4_16k_224-264/2k_MajorTOM224256/checkpoint_14.pth", auto_resume=True, start_epoch=15, num_workers=5, pin_mem=True, find_unused_params=True, log_wandb=True, wandb_project=None, wandb_entity="erik-scheurer", wandb_run_name="2048_2k_MajorTOM224-256", wandb_tags=[], show_user_warnings=False, world_size=32, local_rank=-1, dist_on_itp=False, dist_url="env://", run_name="tokenization/divae/rgb/Sentinel1-ViTB-UNetP4_16k_224-264", patch_size_dec=4, clip_sample=True, epoch_eval=False, config_path="/p/project/geofm4eo/scheurer2/4m4eo/cfgs/default/tokenization/divae/rgb/Sentinel1-ViTB-UNetP4_16k_224-264.yaml", rank=0, gpu=0, distributed=True, dist_backend="nccl", num_tasks=32, all_domains=["sen1@264"], input_size=256, lr=0.0001414213562373095, effective_batch_size=2048)
    """

    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_s1rtc",
        image_size=256,
        n_channels=2,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=kwargs.pop("clip_sample", True),
        **kwargs,
    )

    return tokenizer


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_tokenizer_s1grd(**kwargs):
    """
    S1GRD Tokenizer for TerraMind v1.

    model args: Namespace(patch_size=16, input_size_min=224, input_size_max=256, resolution_step=32, input_size_enc=None, encoder_type="vit_b_enc", decoder_type="unet_patched", post_mlp=True, encoder_ckpt=None, full_ckpt=None, freeze_enc=False, from_scratch=True, dec_transformer_dropout=0.2, quantizer_type="fsq", codebook_size="8-8-8-6-5", latent_dim=5, norm_codes=True, norm_latents=False, codebook_weight=1.0, quantizer_ema_decay=0.99, coef_ema_dead_code=32.0, code_replacement_policy="batch_random", commitment_weight=1.0, kmeans_init=False, num_train_timesteps=1000, prediction_type="sample", beta_schedule="linear", zero_terminal_snr=True, cls_free_guidance_dropout=0.0, masked_cfg=False, masked_cfg_low=0, masked_cfg_high=None, thresholding=True, loss_fn="mse", conditioning="concat", resolution_cond=False, eval_res_cond=None, batch_size=1, batch_size_eval=None, epochs=100, save_ckpt_freq=1, opt="adamw", opt_eps=1e-08, opt_betas=[0.9, 0.99], clip_grad=1.0, skip_grad=None, momentum=0.9, weight_decay=0.05, weight_decay_end=0.05, blr=0.004266666667, warmup_lr=1e-06, min_lr=0.0, warmup_epochs=1, warmup_steps=-1, dtype="fp16", model_ema=True, model_ema_decay=0.9999, model_ema_force_cpu=False, model_ema_update_freq=1, hflip=0.5, domain="sen1@264", mask_value=None, data_path="../fm-geospatial/data/SSL4EOS12/train/", eval_data_path="../fm-geospatial/data/SSL4EOS12/train/", imagenet_default_mean_and_std=False, standardize_surface_normals=False, min_crop_scale=0.8, cache_datasets=False, use_wds=False, s3_endpoint="", s3_data_endpoint=None, wds_n_repeats=1, wds_shuffle_buffer_tar=1000, wds_shuffle_buffer_repeat=1000, s3_multipart_chunksize_mb=512, s3_multipart_threshold_mb=512, dataset_size=None, sample_sentinel_data=True, num_seasons=1, num_locations=64, dist_eval=True, step_eval=True, eval_noise_schedule="DDIMScheduler", num_eval_timesteps=50, input_size_eval=[256], num_eval_metrics_samples=5, eval_freq=5000, eval_metrics_freq=5000, eval_image_log_freq=5000, num_logged_images=15, eval_only=False, no_inception=False, log_codebook_usage=True, output_dir="../fm-geospatial/4m4eo-checkpoints/S1-SSL4EO-full-run/", device="cuda", seed=0, resume="", auto_resume=False, start_epoch=0, num_workers=10, pin_mem=True, find_unused_params=True, log_wandb=True, wandb_project="4m4eo", wandb_entity=None, wandb_run_name="divae/rgb/Sentinel1-ViTB-UNetP4_16k_224-264", wandb_tags=[], show_user_warnings=False, world_size=6, local_rank=-1, dist_on_itp=False, dist_url="env://", s3_path="", s3_save_dir="", run_name="tokenization/divae/rgb/Sentinel1-ViTB-UNetP4_16k_224-264", patch_size_dec=4, clip_sample=True, epoch_eval=False, config_path="cfgs/default/tokenization/divae/rgb/Sentinel1-ViTB-UNetP4_16k_224-264.yaml", rank=0, gpu=0, distributed=True, dist_backend="nccl", num_tasks=6, all_domains=["sen1@264"], input_size=256, lr=0.00010000000000781251)
    """

    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_s1grd",
        image_size=256,
        n_channels=2,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=kwargs.pop("clip_sample", True),
        **kwargs,
    )

    return tokenizer


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_tokenizer_dem(**kwargs):
    """
    DEM Tokenizer for TerraMind v1.

    model args: Namespace(patch_size=16, input_size_min=256, input_size_max=256, resolution_step=32, input_size_enc=None, encoder_type="vit_b_enc", decoder_type="unet_patched", post_mlp=True, encoder_ckpt=None, full_ckpt=None, freeze_enc=False, from_scratch=True, dec_transformer_dropout=0.2, quantizer_type="fsq", codebook_size="8-8-8-6-5", latent_dim=5, norm_codes=True, norm_latents=False, codebook_weight=1.0, quantizer_ema_decay=0.99, coef_ema_dead_code=32.0, code_replacement_policy="batch_random", commitment_weight=1.0, kmeans_init=False, num_train_timesteps=1000, prediction_type="sample", beta_schedule="linear", zero_terminal_snr=True, cls_free_guidance_dropout=0.0, masked_cfg=False, masked_cfg_low=0, masked_cfg_high=None, thresholding=True, loss_fn="mse", conditioning="concat", resolution_cond=False, eval_res_cond=None, batch_size=1, accum_grad=2, batch_size_eval=None, epochs=20, stop_after_epoch=250, save_ckpt_freq=1, opt="adamw", opt_eps=1e-08, opt_betas=[0.9, 0.99], clip_grad=1.0, skip_grad=None, momentum=0.9, weight_decay=0.05, weight_decay_end=0.05, blr=0.0064, warmup_lr=1e-06, min_lr=0.0, warmup_epochs=1, warmup_steps=-1, dtype="fp16", model_ema=True, model_ema_decay=0.9999, model_ema_force_cpu=False, model_ema_update_freq=1, hflip=0.5, domain="dem@264", mask_value=None, data_path="./data/TerraMesh/train", eval_data_path="./data/TerraMesh/val", imagenet_default_mean_and_std=False, standardize_surface_normals=False, min_crop_scale=0.8, cache_datasets=False, dataset_size=None, sample_sentinel_data=False, num_seasons=1, num_locations=64, dist_eval=True, step_eval=True, eval_noise_schedule="DDIMScheduler", num_eval_timesteps=50, input_size_eval=[256], num_eval_metrics_samples=5, eval_freq=2000, eval_metrics_freq=0, eval_image_log_freq=2000, num_logged_images=5, eval_only=False, no_inception=False, log_codebook_usage=True, output_dir="artifacts/runs/sm-dem-20-epochs", device="cuda", seed=0, resume="artifacts/runs/sm-dem-20-epochs/checkpoint_12.pth", auto_resume=True, start_epoch=13, num_workers=8, pin_mem=True, find_unused_params=True, log_wandb=True, wandb_project="terramind-dem", wandb_entity="fasteo-ibm-jsc", wandb_run_name="2048_epochs_20/divae/rgb/DEM-ViTB-UNetP4_16k_224-264", wandb_run_id=None, wandb_tags=[], show_user_warnings=False, world_size=16, local_rank=-1, dist_on_itp=False, dist_url="env://", run_name="tokenization/divae/rgb/DEM-ViTB-UNetP4_16k_224-264", patch_size_dec=4, clip_sample=True, epoch_eval=False, config_path="/p/home/jusers/maurogiovanni1/juwels/4m4eo/cfgs/default/tokenization/divae/rgb/DEM-ViTB-UNetP4_16k_224-264.yaml", rank=0, gpu=0, distributed=True, dist_backend="nccl", num_tasks=16, all_domains=["dem@264"], input_size=256, lr=0.0001414213562373095, effective_batch_size=2048)
    """

    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_dem",
        image_size=256,
        n_channels=1,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=kwargs.pop("clip_sample", True),
        **kwargs,
    )

    return tokenizer


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_tokenizer_lulc(**kwargs):
    """
    LULC Tokenizer for TerraMind v1.

    model args: Namespace(patch_size=16, input_size_min=224, input_size_max=256, resolution_step=16, input_size_enc=None, input_size_dec=None, encoder_type="vit_b_enc", decoder_type="vit_b_dec", post_mlp=True, encoder_ckpt=None, full_ckpt=None, freeze_enc=False, loss_fn="cross_entropy", percept_loss_type=None, percept_loss_blocks="blocks.2-blocks.5-blocks.8-blocks.11", percept_loss_distance="cosine", percept_loss_weight=0.0, quantizer_type="fsq", codebook_size="7-5-5-5-5", num_codebooks=1, latent_dim=5, norm_codes=True, norm_latents=False, codebook_weight=1.0, quantizer_ema_decay=0.99, coef_ema_dead_code=32.0, code_replacement_policy="batch_random", commitment_weight=1.0, kmeans_init=False, batch_size=1, accum_grad=2, batch_size_eval=None, epochs=20, save_ckpt_freq=1, opt="adamw", opt_eps=1e-08, opt_betas=[0.9, 0.99], clip_grad=1.0, skip_grad=None, momentum=0.9, weight_decay=0.05, weight_decay_end=0.05, blr=0.0016, warmup_lr=1e-06, min_lr=0.0, warmup_epochs=1, warmup_steps=-1, dtype="bf16", dtype_percept=None, model_ema=True, model_ema_decay=0.99, model_ema_force_cpu=False, model_ema_update_freq=1, hflip=0.5, domain="lulc@264", mask_value=None, data_path="./data/TerraMesh/train", eval_data_path="./data/TerraMesh/val", min_crop_scale=0.8, cache_datasets=False, num_locations=64, dist_eval=True, step_eval=True, input_size_eval=[256], num_eval_metrics_samples=50000, eval_freq=2000, eval_metrics_freq=0, eval_image_log_freq=1000, num_logged_images=5, eval_only=False, no_inception=False, log_codebook_usage=True, keep_batches_ratio=1.0, output_dir="/p/home/jusers/maurogiovanni1/juwels/4m4eo/artifacts/runs/sm-lulc-20-00016", device="cuda", seed=0, resume="/p/home/jusers/maurogiovanni1/juwels/4m4eo/artifacts/runs/sm-lulc-20-00016/checkpoint_13.pth", auto_resume=True, start_epoch=14, num_workers=8, pin_mem=True, find_unused_params=False, log_wandb=True, wandb_project="terramind-lulc", wandb_entity="fasteo-ibm-jsc", wandb_run_name="2048_epochs_20/lr_00016/acc/vqvae/lulc/LULC_ViTB-ViTB_4k_224-256", wandb_tags=[], show_user_warnings=False, world_size=16, local_rank=-1, dist_on_itp=False, dist_url="env://", run_name="tokenization/vqvae/lulc/LULC_ViTB-ViTB_4k_224-256", stop_after_epoch=250, epoch_eval=False, imagenet_default_mean_and_std=False, sample_sentinel_data=True, num_seasons=1, config_path="/p/home/jusers/maurogiovanni1/juwels/4m4eo/cfgs/default/tokenization/vqvae/lulc/LULC_ViTB-ViTB_4k_224-256.yaml", rank=0, gpu=0, distributed=True, dist_backend="nccl", num_tasks=16, all_domains=["lulc@264"], input_size=256, lr=3.535533905932738e-05, effective_batch_size=2048)
    """

    tokenizer = build_vqvae(
        model_type="vqvae",
        variant="terramind_v1_tokenizer_lulc",
        image_size=256,
        n_channels=10,
        encoder_type="vit_b_enc",
        decoder_type="vit_b_dec",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        quant_type="fsq",
        codebook_size="7-5-5-5-5",
        latent_dim=5,
        **kwargs,
    )

    return tokenizer


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v1_tokenizer_ndvi(**kwargs):
    """
    NDVI Tokenizer for TerraMind v1.

    model args: Namespace(patch_size=16, input_size_min=256, input_size_max=256, resolution_step=32, input_size_enc=None, encoder_type="vit_b_enc", decoder_type="unet_patched", post_mlp=True, encoder_ckpt=None, full_ckpt=None, freeze_enc=False, from_scratch=True, dec_transformer_dropout=0.2, quantizer_type="fsq", codebook_size="8-8-8-6-5", latent_dim=5, norm_codes=True, norm_latents=False, codebook_weight=1.0, quantizer_ema_decay=0.99, coef_ema_dead_code=32.0, code_replacement_policy="batch_random", commitment_weight=1.0, kmeans_init=False, num_train_timesteps=1000, prediction_type="sample", beta_schedule="linear", zero_terminal_snr=True, cls_free_guidance_dropout=0.0, masked_cfg=False, masked_cfg_low=0, masked_cfg_high=None, thresholding=True, loss_fn="mse", conditioning="concat", resolution_cond=False, eval_res_cond=None, batch_size=1, accum_grad=2, batch_size_eval=None, epochs=20, stop_after_epoch=250, save_ckpt_freq=1, opt="adamw", opt_eps=1e-08, opt_betas=[0.9, 0.99], clip_grad=1.0, skip_grad=None, momentum=0.9, weight_decay=0.05, weight_decay_end=0.05, blr=0.0064, warmup_lr=1e-06, min_lr=0.0, warmup_epochs=1, warmup_steps=-1, dtype="fp16", model_ema=True, model_ema_decay=0.9999, model_ema_force_cpu=False, model_ema_update_freq=1, hflip=0.5, domain="ndvi@264", mask_value=None, data_path="./data/TerraMesh/train", eval_data_path="./data/TerraMesh/val", imagenet_default_mean_and_std=False, standardize_surface_normals=False, min_crop_scale=0.8, cache_datasets=False, dataset_size=None, sample_sentinel_data=False, num_seasons=1, num_locations=64, dist_eval=True, step_eval=True, eval_noise_schedule="DDIMScheduler", num_eval_timesteps=50, input_size_eval=[256], num_eval_metrics_samples=5, eval_freq=2000, eval_metrics_freq=0, eval_image_log_freq=2000, num_logged_images=5, eval_only=False, no_inception=False, log_codebook_usage=True, output_dir="artifacts/runs/sm-ndvi-20-epochs", device="cuda", seed=0, resume="artifacts/runs/sm-ndvi-20-epochs/checkpoint_12.pth", auto_resume=True, start_epoch=13, num_workers=8, pin_mem=True, find_unused_params=True, log_wandb=True, wandb_project="terramind-tokenizers", wandb_entity="fasteo-ibm-jsc", wandb_run_name="2048_epochs_20/divae/rgb/NDVI-ViTB-UNetP4_16k_224-264", wandb_run_id=None, wandb_tags=[], show_user_warnings=False, world_size=16, local_rank=-1, dist_on_itp=False, dist_url="env://", run_name="tokenization/divae/rgb/NDVI-ViTB-UNetP4_16k_224-264", patch_size_dec=4, clip_sample=True, epoch_eval=False, config_path="/p/home/jusers/maurogiovanni1/juwels/4m4eo/cfgs/default/tokenization/divae/rgb/NDVI-ViTB-UNetP4_16k_224-264.yaml", rank=0, gpu=0, distributed=True, dist_backend="nccl", num_tasks=16, all_domains=["ndvi@264"], input_size=256, lr=0.0001414213562373095, effective_batch_size=2048)
    """

    tokenizer = build_vqvae(
        variant="terramind_v1_tokenizer_ndvi",
        image_size=256,
        n_channels=1,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=kwargs.pop("clip_sample", True),
        **kwargs,
    )

    return tokenizer


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v01_tokenizer_s2l2a(**kwargs):
    """
    S2L2A Tokenizer for TerraMind v0.1.

    This tokenizer was pre-trained using the SSL4EO-S12 v1.1 samples.
    """
    if kwargs.get("pretrained", False):
        if not os.getenv("HF_TOKEN", None):
            warnings.warn("TerraMind v0.1 models require a HF_TOKEN with access to model weights.")

    tokenizer = build_vqvae(
        variant="terramind_v01_tokenizer_s2l2a",
        image_size=256,
        n_channels=12,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=kwargs.pop("clip_sample", True),
        **kwargs,
    )

    return tokenizer


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v01_tokenizer_s1grd(**kwargs):
    """
    S1GRD Tokenizer for TerraMind v0.1.

    This tokenizer was pre-trained using the SSL4EO-S12 v1.1 samples.
    """
    if kwargs.get("pretrained", False):
        if not os.getenv("HF_TOKEN", None):
            warnings.warn("TerraMind v0.1 models require a HF_TOKEN with access to model weights.")

    tokenizer = build_vqvae(
        variant="terramind_v01_tokenizer_s1grd",
        image_size=256,
        n_channels=2,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=kwargs.pop("clip_sample", True),
        **kwargs,
    )

    return tokenizer


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v01_tokenizer_dem(**kwargs):
    """
    DEM Tokenizer for TerraMind v0.1.

    This tokenizer was pre-trained on the SSL4EO-S12 v1.1 locations.
    """
    if kwargs.get("pretrained", False):
        if not os.getenv("HF_TOKEN", None):
            warnings.warn("TerraMind v0.1 models require a HF_TOKEN with access to model weights.")

    tokenizer = build_vqvae(
        variant="terramind_v01_tokenizer_dem",
        image_size=256,
        n_channels=1,
        encoder_type="vit_b_enc",
        decoder_type="unet_patched",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        patch_size_dec=4,
        quant_type="fsq",
        codebook_size="8-8-8-6-5",
        latent_dim=5,
        clip_sample=kwargs.pop("clip_sample", True),
        **kwargs,
    )

    return tokenizer


@TERRATORCH_FULL_MODEL_REGISTRY.register
def terramind_v01_tokenizer_lulc(**kwargs):
    """
    LULC Tokenizer for TerraMind v0.1.

    This tokenizer was pre-trained on the SSL4EO-S12 v1.1 locations using the nine DynamicWorld classes,
    not with ESRI LULC as v1.0.
    """
    if kwargs.get("pretrained", False):
        if not os.getenv("HF_TOKEN", None):
            warnings.warn("TerraMind v0.1 models require a HF_TOKEN with access to model weights.")

    tokenizer = build_vqvae(
        model_type="vqvae",
        variant="terramind_v01_tokenizer_lulc",
        image_size=256,
        n_channels=9,
        encoder_type="vit_b_enc",
        decoder_type="vit_b_dec",
        prediction_type="sample",
        post_mlp=True,
        patch_size=16,
        quant_type="fsq",
        codebook_size="7-5-5-5-5",
        latent_dim=5,
        **kwargs,
    )

    return tokenizer


def terramind_v01_caption_tokenizer(pretrained=True, tokenizer_file=None, *args, **kwargs):
    if not tokenizers_available:
        warnings.warn("Cannot import tokenizers. \nMake sure to install `pip install tokenizers`.")
        raise import_error_tokenizers

    if pretrained and tokenizer_file is None:
        tokenizer_file = hf_hub_download(
            repo_id=pretrained_weights["terramind_v01_caption_tokenizer"]["hf_hub_id"],
            filename=pretrained_weights["terramind_v01_caption_tokenizer"]["hf_hub_filename"],
        )

    return CaptionTokenizer(tokenizer_file=tokenizer_file, *args, **kwargs)


def terramind_v1_coords_tokenizer(pretrained=True, tokenizer_file=None, *args, **kwargs):
    if not tokenizers_available:
        warnings.warn("Cannot import tokenizers. \nMake sure to install `pip install tokenizers`.")
        raise import_error_tokenizers

    if pretrained and tokenizer_file is None:
        tokenizer_file = hf_hub_download(
            repo_id=pretrained_weights["terramind_v1_coords_tokenizer"]["hf_hub_id"],
            filename=pretrained_weights["terramind_v1_coords_tokenizer"]["hf_hub_filename"],
        )

    return CoordsTokenizer(tokenizer_file=tokenizer_file, *args, **kwargs)
