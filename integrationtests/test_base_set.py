import gc
import glob
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import rasterio
import requests
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from terratorch.cli_tools import LightningInferenceModel
from terratorch.datamodules import HelioNetCDFDataModule
from terratorch.datasets import HelioNetCDFDataset
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.tasks import InferenceTask

# Allow overriding default test asset locations via environment variables.
# DATA_ROOT is the shared storage prefix that all the default paths below live
# under. Overriding it (e.g. when the /dccstor mount is unavailable) relocates the
# input datasets referenced by the fit configs; it defaults to "/dccstor" so the
# behaviour is unchanged when unset.
DATA_ROOT = os.getenv("TERRATORCH_DATA_ROOT", "/dccstor")
TEST_MODELS_ROOT = os.getenv(
    "TERRATORCH_TEST_MODELS_ROOT",
    f"{DATA_ROOT}/terratorch/shared/integrationtests/testing_models",
)
TMP_ROOT = os.getenv("TERRATORCH_TMP_ROOT", f"{DATA_ROOT}/terratorch/tmp")
TEST_CHECKPOINTS_ROOT = os.getenv("TERRATORCH_TEST_CHECKPOINTS_ROOT", TEST_MODELS_ROOT)


def free_gpu():
    try:
        subprocess.run(["nvidia-smi", "--gpu-reset", "-i", "0"], check=True)
    except Exception as e:
        print("GPU reset failed:", e)


MODEL_CONFIG_MAP = {
    "eo_v1_100": "test_encoderdecoder_eo_v1_100_model_factory.yaml",
    "eo_v2_300": "test_encoderdecoder_eo_v2_300_model_factory.yaml",
    "eo_v2_600": "test_encoderdecoder_eo_v2_600_model_factory.yaml",
    "swinb": "test_prithvi_swinB_model_factory_config.yaml",
    "swinl": "test_prithvi_swinL_model_factory_config.yaml",
    "smp_resnet34": "test_smp_resnet34_model_factory_config.yaml",
    "timm_resnet34": "test_encoderdecoder_timm_resnet34_model_factory.yaml",
    "timm_resnet18": "test_encoder_decoder_timm_resnet18_model_factory.yaml",
    "timm_resnet50": "test_encoder_decoder_timm_resnet50_model_factory.yaml",
    "timm_resnet101": "test_encoder_decoder_timm_resnet101_model_factory.yaml",
    "timm_resnet152": "test_encoder_decoder_timm_resnet152_model_factory.yaml",
    "clay_v1": "test_encoderdecoder_clay_v1_base_model_factory.yaml",
    "timm_convnext_large": "test_encoderdecoder_timm_convnext_large-fb-in22k_model_factory.yaml",
    "timm_convnext_xlarge": "test_encoderdecoder_timm_convnext_xlarge-fb-in22k_model_factory.yaml",
    "terramind_large": "test_terramind_large.yaml",
    "terramind_base": "test_terramind_base.yaml",
}


@pytest.mark.parametrize(
    "model_name",
    [
        "eo_v1_100",
        "eo_v2_300",
        "eo_v2_600",
        "swinb",
        "swinl",
        "smp_resnet34",
        "timm_resnet34",
        "timm_resnet18",
        "timm_resnet50",
        "timm_resnet101",
        "timm_resnet152",
        "clay_v1",
        "timm_convnext_large",
        "timm_convnext_xlarge",
        "terramind_base",
        "terramind_large",
    ],
)
def test_models_fit(model_name):
    import tempfile
    import yaml

    # Read the original config file
    config_path = f"./integrationtests/configs/{MODEL_CONFIG_MAP[model_name]}"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Relocate hard-coded storage paths in the config to the configured roots.
    # Replace the output/checkpoint dir first (most specific), then relocate any
    # remaining /dccstor input-dataset paths onto DATA_ROOT. Both are no-ops with
    # the default settings, so the shared-cluster behaviour is unchanged.
    config_str = yaml.dump(config)
    config_str = config_str.replace("/dccstor/terratorch/tmp", TMP_ROOT)
    config_str = config_str.replace("/dccstor", DATA_ROOT)
    config = yaml.safe_load(config_str)

    # Write the modified config to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp_config:
        yaml.dump(config, tmp_config)
        tmp_config.flush()  # Ensure content is written to disk
        tmp_config_path = tmp_config.name

        result = subprocess.run(
            ["terratorch", "fit", "-c", tmp_config_path],
            capture_output=True,
            text=True,
            check=False,
        )
        torch.cuda.empty_cache()
        free_gpu()

        # Print the captured output
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Check the return code
        assert result.returncode == 0, (
            f"Test failed with return code {result.returncode}STDOUT: {result.stdout}STDERR: {result.stderr}"
        )

    gc.collect()


def download_and_open_tiff(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        pytest.fail(f"Failed to download TIFF image from URL: {url} (Status code: {response.status_code})")

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def create_deploy_config(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No such file: {config_path}")

    dir_name = os.path.dirname(config_path)
    name, ext = os.path.splitext(os.path.basename(config_path))
    deploy_config_path = os.path.join(dir_name, f"{name}_deploy{ext}")

    shutil.copyfile(config_path, deploy_config_path)

    return deploy_config_path


def update_grep_config_in_file(config_path: str, new_img_pattern: str):
    """Function to update img_grep in config

    Parameters
    ----------
    config_path : str
        Config file path
    new_img_pattern : str
        New img_grep pattern
    """

    with open(config_path) as file:
        config = file.read()

    # Find the current img_grep pattern (this assumes there is one img_grep line)
    current_img_pattern_match = re.search(r"img_grep:\s*'(.*?)'", config)

    # If the img_grep line exists, update it with the new img_pattern
    if current_img_pattern_match:
        config = re.sub(r"img_grep:\s*'.*'", f"img_grep: '{new_img_pattern}'", config)

    # Write the updated config back to the file
    with open(config_path, "w") as file:
        file.write(config)


@pytest.fixture(scope="session")
def buildings_image(tmp_path_factory):
    url = "https://ibm.box.com/shared/static/fzt9ng3si2qk3l7b4iz6u9egfuj9re4o.tif"
    temp_dir = tmp_path_factory.mktemp("data")
    local_path = temp_dir / "buildings-img.tiff"

    download_and_open_tiff(url=url, dest_path=local_path)

    return str(local_path)


@pytest.fixture(scope="session")
def burnscars_image(tmp_path_factory):
    url = " https://s3.us-east.cloud-object-storage.appdomain.cloud/geospatial-studio-example-data/examples-for-inference/park_fire_scaled.tif"
    temp_dir = tmp_path_factory.mktemp("data")
    local_path = temp_dir / "burnscars_merged.tif"

    download_and_open_tiff(url=url, dest_path=local_path)

    return str(local_path)


@pytest.fixture(scope="session")
def floods_image_dir(tmp_path_factory):
    """Download and extract the floods image into temp directory."""
    url = "https://s3.us-east.cloud-object-storage.appdomain.cloud/geospatial-studio-example-data/examples-for-inference/montenegro-brazil-floods-20231120-S1L2A.wgs84.tif"
    temp_dir = tmp_path_factory.mktemp("data")
    local_path: Any = temp_dir / "floods_image.tif"

    download_and_open_tiff(url=url, dest_path=local_path)
    with rasterio.open(local_path) as src:
        # Get metadata
        profile = src.profile.copy()

        s2_bands = list(range(1, 7))  # Bands 1-6 (rasterio is 1-indexed)
        s2_data = src.read(s2_bands)

        # Use bands 1-6, then repeat them to get to 13 bands
        s2_extended = []
        for i in range(13):
            band_idx = i % 6
            s2_extended.append(s2_data[band_idx])
        s2_extended = np.array(s2_extended)

        s2_output = temp_dir / "flood_image_S2Hand.tif"
        profile.update(count=13)
        with rasterio.open(s2_output, "w", **profile) as dst:
            dst.write(s2_extended)

        # S1 needs 2 bands - use last band duplicated
        s1_band = src.read(7)  # Last band
        s1_data = np.array([s1_band, s1_band])  # Duplicate to get 2 bands

        s1_output = temp_dir / "flood_image_S1Hand.tif"
        profile.update(count=2)
        with rasterio.open(s1_output, "w", **profile) as dst:
            dst.write(s1_data)

    return str(temp_dir)


def run_inference(config, checkpoint, image):
    model = LightningInferenceModel.from_config(config_path=config, checkpoint_path=checkpoint)

    predictions = model.inference(image)

    return predictions


def run_inference_on_dir(config, checkpoint, image_dir):
    model = LightningInferenceModel.from_config(config_path=config, checkpoint_path=checkpoint)

    predictions, file_names = model.inference_on_dir(image_dir)

    return predictions, file_names


@pytest.mark.parametrize(
    "model_name",
    ["eo_v1_100", "eo_v2_300", "eo_v2_600", "smp_resnet34_model_factory", "encoderdecoder_resnet34_model_factory"],
)
def test_legacy_buildings_predict(buildings_image, model_name):
    # Models trained with an earlier terratorch version
    model_dir = os.path.join(TEST_MODELS_ROOT, f"buildings_{model_name}")
    config_path = os.path.join(model_dir, f"config_{model_name}.yaml")
    checkpoint_dir = os.path.join(TEST_CHECKPOINTS_ROOT, f"buildings_{model_name}")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.ckpt")

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=buildings_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


@pytest.mark.parametrize("model_name", ["terramind_base"])
def test_legacy_floods_predict(floods_image_dir, model_name):
    # Terramind models trained with tt version==1.1
    model_dir = os.path.join(TEST_MODELS_ROOT, f"floods_{model_name}")
    config_path = os.path.join(model_dir, f"config_{model_name}.yaml")
    checkpoint_dir = os.path.join(TEST_CHECKPOINTS_ROOT, f"floods_{model_name}")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.ckpt")

    preds, file_names = run_inference_on_dir(config=config_path, checkpoint=checkpoint_path, image_dir=floods_image_dir)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


@pytest.mark.parametrize("model_name", ["swinb", "swinl"])
def test_legacy_burnscars_predict(burnscars_image, model_name):
    # Models trained with an earlier terratorch version
    model_dir = os.path.join(TEST_MODELS_ROOT, f"burnscars_{model_name}")
    config_path = os.path.join(model_dir, f"config_{model_name}.yaml")
    checkpoint_dir = os.path.join(TEST_CHECKPOINTS_ROOT, f"burnscars_{model_name}")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{model_name}.ckpt")

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=burnscars_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


def test_surya():
    """
    Additional test for the Surya model
    """
    if not os.path.isdir("experiment"):
        os.mkdir("experiment")

    root_dir = os.path.join(TMP_ROOT, "experiment")
    # Downloading validation data
    snapshot_download(repo_id="nasa-ibm-ai4science/Surya-1.0_validation_data", repo_type="dataset", local_dir=root_dir)

    hf_hub_download(repo_id="nasa-ibm-ai4science/Surya-1.0", filename="scalers.yaml", local_dir=root_dir)

    # Creating index file
    sample_files = glob.glob(os.path.join(root_dir, "*.nc"))
    paths = sorted(sample_files)
    present = len(paths) * [1]

    timestamps = []
    for ff in paths:
        filename = os.path.basename(ff).replace(".nc", "")
        date, timestamp = filename.split("_")

        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:])
        hour = int(timestamp[:2])
        minutes = int(timestamp[2:])
        seconds = 0

        date_datetime = datetime(year, month, day, hour, minutes, seconds).strftime("%Y-%m-%d %H:%M:%S")

        timestamps.append(date_datetime)

    index_dict = {"path": paths, "timestep": timestamps, "present": present}
    index_dataframe = pd.DataFrame(index_dict)
    index_dataframe.to_csv(os.path.join(root_dir, "index.csv"))
    index_path = os.path.join(root_dir, "index.csv")
    scalers_path = os.path.join(root_dir, "scalers.yaml")

    channels = [
        "aia94",
        "aia131",
        "aia171",
        "aia193",
        "aia211",
        "aia304",
        "aia335",
        "aia1600",
        "hmi_m",
        "hmi_bx",
        "hmi_by",
        "hmi_bz",
        "hmi_v",
    ]

    run = "predict"

    start_time = time.time()
    if run == "predict":
        datamodule = HelioNetCDFDataModule(
            train_index_path=index_path,
            test_index_path=index_path,
            val_index_path=index_path,
            predict_index_path=index_path,
            batch_size=1,
            num_workers=0,
            time_delta_input_minutes=[-60, 0],
            time_delta_target_minutes=+60,
            channels=channels,
            n_input_timestamps=2,
            rollout_steps=1,
            scalers=scalers_path,
        )

        model_name = "heliofm_backbone_surya"
        backbone = BACKBONE_REGISTRY.build(model_name, pretrained=True)
        datamodule.setup("predict")
        model = InferenceTask(model=backbone)
        pl.seed_everything(0)
        # Lightning Trainer
        trainer = pl.Trainer(
            accelerator="gpu",
            strategy="auto",
            devices=1,
            # precision='bf16-mixed',
            num_nodes=1,
            logger=True,
            max_epochs=1,
            log_every_n_steps=1,
            enable_checkpointing=True,
            callbacks=[pl.callbacks.RichProgressBar()],
            default_root_dir="output/heliofm",
        )

        # Training
        prediction = trainer.predict(model, datamodule=datamodule)
        print(f"Elapsed time: {time.time() - start_time} s")

        assert all([isinstance(p, torch.Tensor) for p in prediction]), (
            f"Expected predictions to be type torch.Tensor, got {type(prediction)}"
        )


## Only run these tests after running test_finetune.py.
## Uses the recently created checkpoints to test if the
## current terratorch version runs inference successfully


@pytest.mark.parametrize("config_name", ["smp_resnet34", "timm_resnet34"])
def test_latest_terratorch_version_buildings_predict(config_name, buildings_image):
    # Models trained with current terratorch version
    config_path = os.path.join(TMP_ROOT, config_name, "lightning_logs", "version_0", "config_deploy.yaml")

    pattern = os.path.join(TMP_ROOT, config_name, "best-state_dict-epoch=*.ckpt")
    checkpoint_path = glob.glob(pattern)[0]

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=buildings_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


@pytest.mark.parametrize("config_name", ["terramind_base", "terramind_large"])
def test_latest_terratorch_version_floods_predict(config_name, floods_image_dir):
    # Models trained with current terratorch version
    config_path = os.path.join(TMP_ROOT, config_name, "lightning_logs", "version_0", "config_deploy.yaml")

    pattern = os.path.join(TMP_ROOT, config_name, "best-state_dict-epoch=*.ckpt")
    checkpoint_path = glob.glob(pattern)[0]

    preds, file_names = run_inference_on_dir(config=config_path, checkpoint=checkpoint_path, image_dir=floods_image_dir)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


@pytest.mark.parametrize(
    "config_name",
    [
        "eo_v1_100",
        "eo_v2_300",
        "eo_v2_600",
        "swinb",
        "swinl",
        "timm_resnet18",
        "timm_resnet50",
        "timm_resnet101",
        "timm_resnet152",
        "clay_v1",
        "timm_convnext_large",
        "timm_convnext_xlarge",
    ],
)
# Models trained with current terratorch version
def test_latest_terratorch_version_burnscars_predict(config_name, burnscars_image):
    # config_path = f"configs/test_{config_name}.yaml"
    config_path = os.path.join(TMP_ROOT, config_name, "lightning_logs", "version_0", "config_deploy.yaml")

    pattern = os.path.join(TMP_ROOT, config_name, "best-state_dict-epoch=*.ckpt")
    checkpoint_path = glob.glob(pattern)[0]

    preds = run_inference(config=config_path, checkpoint=checkpoint_path, image=burnscars_image)

    assert isinstance(preds, torch.Tensor), f"Expected predictions to be type torch.Tensor, got {type(preds)}"

    gc.collect()


@pytest.mark.parametrize(
    "model_name",
    [
        "eo_v1_100",
        "eo_v2_300",
        "eo_v2_600",
        "swinb",
        "swinl",
        "smp_resnet34",
        "timm_resnet34",
        "timm_resnet18",
        "timm_resnet50",
        "timm_resnet101",
        "timm_resnet152",
        "clay_v1",
        "timm_convnext_large",
        "timm_convnext_xlarge",
        "terramind_large",
        "terramind_base",
        "experiment",
    ],
)
def test_cleanup(model_name):
    # Delete all folders creating during finetuning after running inference.
    full_path = os.path.join(TMP_ROOT, model_name)
    print("Attempting to delete:", full_path)
    try:
        shutil.rmtree(full_path)
        print(f"Deleted: {full_path}")
    except FileNotFoundError:
        print(f"Already deleted or missing: {full_path}")
    except PermissionError:
        print(f"Permission denied: {full_path}")
    except Exception as e:
        print(f"Error deleting {full_path}: {e}")

    assert not os.path.exists(full_path)

    gc.collect()
