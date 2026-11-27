from pathlib import Path
import gc
import pytest
from utils import create_dummy_tiff


@pytest.fixture
def dummy_multimodal_data(tmp_path: Path) -> Path:
    root = tmp_path / "generic_multimodal"
    root.mkdir(parents=True, exist_ok=True)
    train_data_dir = root / "train" / "input"
    val_data_dir = root / "val" / "input"
    test_data_dir = root / "test" / "input"
    predict_data_dir = root / "predict" / "input"
    for p in [train_data_dir, val_data_dir, test_data_dir, predict_data_dir]:
        p.mkdir(parents=True, exist_ok=True)
    train_label_dir = root / "train" / "labels"
    val_label_dir = root / "val" / "labels"
    test_label_dir = root / "test" / "labels"
    for p in [train_label_dir, val_label_dir, test_label_dir]:
        p.mkdir(parents=True, exist_ok=True)
    shape_img = (16, 16, 3)
    shape_mask = (16, 16, 1)
    pixel_values_image = [0, 50, 100, 200]
    pixel_values_label = [0, 1]

    create_dummy_tiff(
        path=str(train_data_dir / "sample1_mod1.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(val_data_dir / "sample1_mod1.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(test_data_dir / "sample1_mod1.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(predict_data_dir / "sample1_mod1.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(train_data_dir / "sample1_mod2.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(val_data_dir / "sample1_mod2.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(test_data_dir / "sample1_mod2.tif"),
        shape=shape_img,
        pixel_values=pixel_values_image
    )
    create_dummy_tiff(
        path=str(train_label_dir / "sample1.tif"),
        shape=shape_mask,
        pixel_values=pixel_values_label
    )
    create_dummy_tiff(
        path=str(val_label_dir / "sample1.tif"),
        shape=shape_mask,
        pixel_values=pixel_values_label
    )
    create_dummy_tiff(
        path=str(test_label_dir / "sample1.tif"),
        shape=shape_mask,
        pixel_values=pixel_values_label
    )

    return root


def test_generic_multimodal_datamodule(dummy_multimodal_data: Path):
    from terratorch.datamodules.generic_multimodal_data_module import GenericMultiModalDataModule

    train_data_root = {"mod1": str(dummy_multimodal_data / "train" / "input")}
    val_data_root = {"mod1": str(dummy_multimodal_data / "val" / "input")}
    test_data_root = {"mod1": str(dummy_multimodal_data / "test" / "input")}
    predict_data_root = {"mod1": str(dummy_multimodal_data / "predict" / "input")}
    train_label_root = str(dummy_multimodal_data / "train" / "labels")
    val_label_root = str(dummy_multimodal_data / "val" / "labels")
    test_label_root = str(dummy_multimodal_data / "test" / "labels")
    means = {"mod1": [0.0, 0.0, 0.0]}
    stds = {"mod1": [1.0, 1.0, 1.0]}

    dm = GenericMultiModalDataModule(
        modalities=["mod1"],
        task="segmentation",
        num_classes=2,
        batch_size=2,
        train_data_root=train_data_root,
        val_data_root=val_data_root,
        test_data_root=test_data_root,
        predict_data_root=predict_data_root,
        image_grep={"mod1": "*_mod1.tif"},
        label_grep="*tif",
        train_label_data_root=train_label_root,
        val_label_data_root=val_label_root,
        test_label_data_root=test_label_root,
        allow_substring_file_names=True,
        means=means,
        stds=stds,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        concat_bands=True,
    )

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing key 'image' in train batch"
    assert "mask" in train_batch, "Missing key 'mask' in train batch"
    dm.setup("validate")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing key 'image' in validation batch"
    assert "mask" in val_batch, "Missing key 'mask' in validation batch"
    dm.setup("test")
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing key 'image' in test batch"
    assert "mask" in test_batch, "Missing key 'mask' in test batch"
    dm.setup("predict")
    predict_loader = dm.predict_dataloader()
    predict_batch = next(iter(predict_loader))
    assert "image" in predict_batch, "Missing key 'image' in predict batch"

    gc.collect()


def test_generic_multimodal_datamodule_missing_mod(dummy_multimodal_data: Path):
    from terratorch.datamodules.generic_multimodal_data_module import GenericMultiModalDataModule

    train_data_root = {"mod1": str(dummy_multimodal_data / "train" / "input"),
                       "mod2": str(dummy_multimodal_data / "train" / "input")}
    val_data_root = {"mod1": str(dummy_multimodal_data / "val" / "input"),
                     "mod2": str(dummy_multimodal_data / "val" / "input")}
    test_data_root = {"mod1": str(dummy_multimodal_data / "test" / "input"),
                      "mod2": str(dummy_multimodal_data / "test" / "input")}
    predict_data_root = {"mod1": str(dummy_multimodal_data / "predict" / "input"),
                         "mod2": str(dummy_multimodal_data / "predict" / "input")}
    train_label_root = str(dummy_multimodal_data / "train" / "labels")
    val_label_root = str(dummy_multimodal_data / "val" / "labels")
    test_label_root = str(dummy_multimodal_data / "test" / "labels")
    means = {"mod1": [0.0, 0.0, 0.0], "mod2": [0.0, 0.0, 0.0]}
    stds = {"mod1": [1.0, 1.0, 1.0], "mod2": [1.0, 1.0, 1.0]}

    dm = GenericMultiModalDataModule(
        modalities=["mod1", "mod2"],
        task="segmentation",
        num_classes=2,
        batch_size=2,
        train_data_root=train_data_root,
        val_data_root=val_data_root,
        test_data_root=test_data_root,
        predict_data_root=predict_data_root,
        image_grep={"mod1": "*_mod1.tif", "mod2": "*_mod2.tif"},
        train_label_data_root=train_label_root,
        val_label_data_root=val_label_root,
        test_label_data_root=test_label_root,
        allow_missing_modalities=True,
        allow_substring_file_names=True,
        means=means,
        stds=stds,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "mod1" in train_batch["image"], "Missing key 'mod1' in train batch['image']"
    assert "mod2" in train_batch["image"], "Missing key 'mod2' in train batch['image']"
    assert "mask" in train_batch, "Missing key 'mask' in train batch"
    dm.setup("validate")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "mod1" in val_batch["image"], "Missing key 'mod1' in validation batch['image']"
    assert "mod2" in val_batch["image"], "Missing key 'mod2' in validation batch['image']"
    assert "mask" in val_batch, "Missing key 'mask' in validation batch"
    dm.setup("test")
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "mod1" in test_batch["image"], "Missing key 'mod1' in test batch['image']"
    assert "mod2" in test_batch["image"], "Missing key 'mod2' in test batch['image']"
    assert "mask" in test_batch, "Missing key 'mask' in test batch"
    dm.setup("predict")
    predict_loader = dm.predict_dataloader()
    predict_batch = next(iter(predict_loader))
    assert "mod1" in predict_batch["image"], "Missing key 'mod1' in predict batch['image']"
    # mod2 not expected to test allow_missing_modalities

    gc.collect()

def _dict_transform_datamodule(dummy_multimodal_data: Path, val_transform):
    from terratorch.datamodules.generic_multimodal_data_module import GenericMultiModalDataModule

    return GenericMultiModalDataModule(
        modalities=["mod1", "mod2"],
        task="segmentation",
        num_classes=2,
        batch_size=1,
        train_data_root={m: str(dummy_multimodal_data / "train" / "input") for m in ["mod1", "mod2"]},
        val_data_root={m: str(dummy_multimodal_data / "val" / "input") for m in ["mod1", "mod2"]},
        test_data_root={m: str(dummy_multimodal_data / "test" / "input") for m in ["mod1", "mod2"]},
        image_grep={"mod1": "*_mod1.tif", "mod2": "*_mod2.tif"},
        label_grep="*tif",
        train_label_data_root=str(dummy_multimodal_data / "train" / "labels"),
        val_label_data_root=str(dummy_multimodal_data / "val" / "labels"),
        test_label_data_root=str(dummy_multimodal_data / "test" / "labels"),
        allow_substring_file_names=True,
        means={"mod1": [0.0, 0.0, 0.0], "mod2": [0.0, 0.0, 0.0]},
        stds={"mod1": [1.0, 1.0, 1.0], "mod2": [1.0, 1.0, 1.0]},
        num_workers=0,
        val_transform=val_transform,
    )


# A modality missing from the dict (mod2 in the second case) is only converted to a tensor
@pytest.mark.parametrize("transformed_modalities", [["mod1", "mod2"], ["mod1"]])
def test_generic_multimodal_datamodule_dict_transform(dummy_multimodal_data: Path, transformed_modalities):
    import torch
    from albumentations.pytorch import ToTensorV2

    val_transform = {m: [ToTensorV2()] for m in transformed_modalities}
    dm = _dict_transform_datamodule(dummy_multimodal_data, val_transform)
    dm.setup("fit")
    sample = dm.val_dataset[0]
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["mask"].dtype == torch.long
    assert sample["mask"].shape == (16, 16)
    for m in ["mod1", "mod2"]:
        assert isinstance(sample["image"][m], torch.Tensor)
        assert sample["image"][m].shape == (3, 16, 16)
    batch = next(iter(dm.val_dataloader()))
    assert batch["mask"].shape == (1, 16, 16)