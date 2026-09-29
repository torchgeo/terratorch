import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from h5py import special_dtype
from utils import create_dummy_tiff


def create_sen4map_compound_dataset_h5(
    file_path: str,
    dataset_name: str,
    bands: tuple[str, ...],
    shape: tuple[int, int, int] = (12, 256, 256),
    lc1: str = "B10"
) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    compound_dtype = np.dtype([(band, "uint8") for band in bands])

    data = np.zeros(shape, dtype=compound_dtype)
    for band in bands:
        data[band] = 1

    with h5py.File(file_path, "w") as f:
        dset = f.create_dataset(dataset_name, data=data)
        dset.attrs["lc1"] = lc1

        vlstr_dtype = special_dtype(vlen=str)
        image_ids = np.array([f"201801{i+1:02d}" for i in range(12)], dtype=object)
        dset.attrs.create("Image_ID", data=image_ids, dtype=vlstr_dtype)


@pytest.fixture
def dummy_sen4map_h5_data(tmp_path) -> tuple[str, list[str]]:

    expected_bands = ("SCL","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")
    file_path = tmp_path / "sen4map_data.hdf5"
    dataset_name = "regionA_dummy"

    create_sen4map_compound_dataset_h5(
        file_path=str(file_path),
        dataset_name=dataset_name,
        bands=expected_bands,
        shape=(12,256,256),
        lc1="B10"
    )
    keys = [dataset_name]
    return str(file_path), keys

@pytest.fixture
def dummy_sen4map_keys(tmp_path, dummy_sen4map_h5_data) -> dict[str, Path]:
    _, keys = dummy_sen4map_h5_data
    train_path = tmp_path / "train_keys.pkl"
    val_path   = tmp_path / "val_keys.pkl"
    test_path  = tmp_path / "test_keys.pkl"
    for pkl_path in [train_path, val_path, test_path]:
        with open(pkl_path, "wb") as f:
            pickle.dump(keys, f)
    return {
        "train": train_path,
        "val":   val_path,
        "test":  test_path
    }

@pytest.fixture
def dummy_sen4map_data_root(tmp_path, dummy_sen4map_h5_data) -> str:
    root = tmp_path / "sen4map_dummy"
    root.mkdir()

    for split in ["train","val","test"]:
        with open(root / f"{split}.txt","w") as f:
            f.write("regionA_dummy\n")

    region_dir = root / "regionA_dummy"
    images_dir = region_dir / "images"
    labels_dir = region_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    h5path, _ = dummy_sen4map_h5_data
    dest = images_dir / "regionA_dummy"
    os.system(f"cp {h5path} {dest}")

    create_dummy_tiff(str(labels_dir / "regionA_dummy"), (256,256), 2)

    chips_df = pd.DataFrame({
        "chip_id":[1],
        "first_img_date":["20200101"],
        "middle_img_date":["20200115"],
        "last_img_date":["20200131"],
        "merged_label":[0]
    })
    chips_df.to_csv(root / "chips_df.csv", index=False)

    return str(root)

def test_sen4map_datamodule(dummy_sen4map_data_root, dummy_sen4map_keys):
    from terratorch.datamodules import Sen4MapLucasDataModule

    dm = Sen4MapLucasDataModule(
        train_hdf5_path = Path(dummy_sen4map_data_root)/"regionA_dummy"/"images"/"regionA_dummy",
        train_hdf5_keys_path = dummy_sen4map_keys["train"],
        val_hdf5_path = Path(dummy_sen4map_data_root)/"regionA_dummy"/"images"/"regionA_dummy",
        val_hdf5_keys_path = dummy_sen4map_keys["val"],
        test_hdf5_path = Path(dummy_sen4map_data_root)/"regionA_dummy"/"images"/"regionA_dummy",
        test_hdf5_keys_path = dummy_sen4map_keys["test"],
        batch_size=1,
        num_workers=0,
        resize=False,
        prefetch_factor=None,
    )

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    train_batch  = next(iter(train_loader))
    assert "image" in train_batch, "Missing 'image' in train batch"
    assert "label"  in train_batch, "Missing 'mask'  in train batch"
    dm.setup("validate")
    val_loader = dm.val_dataloader()
    val_batch  = next(iter(val_loader))
    assert "image" in val_batch, "Missing 'image' in val batch"
    assert "label"  in val_batch, "Missing 'mask'  in val batch"
    dm.setup("test")
    test_loader = dm.test_dataloader()
    test_batch  = next(iter(test_loader))
    assert "image" in test_batch, "Missing 'image' in test batch"
    assert "label"  in test_batch, "Missing 'mask'  in test batch"


class _Malicious:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


def test_sen4map_rejects_unsafe_keys_file(tmp_path):
    from terratorch.datamodules import Sen4MapLucasDataModule

    marker = tmp_path / "pwned"
    keys_path = tmp_path / "keys.pkl"
    for payload, error in [(_Malicious(str(marker)), pickle.UnpicklingError), ({"key": "value"}, ValueError)]:
        with open(keys_path, "wb") as f:
            pickle.dump(payload, f)
        dm = Sen4MapLucasDataModule(0, 0, train_hdf5_keys_path=keys_path)
        with pytest.raises(error):
            dm.setup("fit")
    assert not marker.exists()
