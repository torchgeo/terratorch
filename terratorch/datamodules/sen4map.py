import pickle

import h5py
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import InterpolationMode

from terratorch.datasets import Sen4MapDatasetMonthlyComposites


class _KeysUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Forbidden global '{module}.{name}' in HDF5 keys file.")


class Sen4MapLucasDataModule(pl.LightningDataModule):
    """NonGeo LightningDataModule implementation for Sen4map."""

    def __init__(
        self,
        batch_size,
        num_workers,
        prefetch_factor=0,
        # dataset_bands:list[HLSBands|int] = None,
        # input_bands:list[HLSBands|int] = None,
        train_hdf5_path=None,
        train_hdf5_keys_path=None,
        test_hdf5_path=None,
        test_hdf5_keys_path=None,
        val_hdf5_path=None,
        val_hdf5_keys_path=None,
        **kwargs,
    ):
        """
        Initializes the Sen4MapLucasDataModule for handling Sen4Map monthly composites.

        Args:
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of worker processes for data loading.
            prefetch_factor (int, optional): Number of samples to prefetch per worker. Defaults to 0.
            train_hdf5_path (str, optional): Path to the training HDF5 file.
            train_hdf5_keys_path (str, optional): Path to the training HDF5 keys file.
            test_hdf5_path (str, optional): Path to the testing HDF5 file.
            test_hdf5_keys_path (str, optional): Path to the testing HDF5 keys file.
            val_hdf5_path (str, optional): Path to the validation HDF5 file.
            val_hdf5_keys_path (str, optional): Path to the validation HDF5 keys file.
            train_hdf5_keys_save_path (str, optional): (from kwargs) Path to save generated train keys.
            test_hdf5_keys_save_path (str, optional): (from kwargs) Path to save generated test keys.
            val_hdf5_keys_save_path (str, optional): (from kwargs) Path to save generated validation keys.
            shuffle (bool, optional): Global shuffle flag.
            train_shuffle (bool, optional): Shuffle flag for training data; defaults to global shuffle if unset.
            val_shuffle (bool, optional): Shuffle flag for validation data.
            test_shuffle (bool, optional): Shuffle flag for test data.
            train_data_fraction (float, optional): Fraction of training data to use. Defaults to 1.0.
            val_data_fraction (float, optional): Fraction of validation data to use. Defaults to 1.0.
            test_data_fraction (float, optional): Fraction of test data to use. Defaults to 1.0.
            all_hdf5_data_path (str, optional): General HDF5 data path for all splits. If provided, overrides specific paths.
            resize (bool, optional): Whether to resize images. Defaults to False.
            resize_to (int or tuple, optional): Target size for resizing images.
            resize_interpolation (str, optional): Interpolation mode for resizing ('bilinear', 'bicubic', etc.).
            resize_antialiasing (bool, optional): Whether to apply antialiasing during resizing. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        self.prepare_data_per_node = False
        self._log_hyperparams = None
        self.allow_zero_length_dataloader_with_multiple_devices = False

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.train_hdf5_path = train_hdf5_path
        self.test_hdf5_path = test_hdf5_path
        self.val_hdf5_path = val_hdf5_path

        self.train_hdf5_keys_path = train_hdf5_keys_path
        self.test_hdf5_keys_path = test_hdf5_keys_path
        self.val_hdf5_keys_path = val_hdf5_keys_path

        if train_hdf5_path and not train_hdf5_keys_path:
            print(
                "Train dataset path provided but not the path to the dataset keys. Generating the keys might take a few minutes."
            )
        if test_hdf5_path and not test_hdf5_keys_path:
            print(
                "Test dataset path provided but not the path to the dataset keys. Generating the keys might take a few minutes."
            )
        if val_hdf5_path and not val_hdf5_keys_path:
            print(
                "Val dataset path provided but not the path to the dataset keys. Generating the keys might take a few minutes."
            )

        self.train_hdf5_keys_save_path = kwargs.pop("train_hdf5_keys_save_path", None)
        self.test_hdf5_keys_save_path = kwargs.pop("test_hdf5_keys_save_path", None)
        self.val_hdf5_keys_save_path = kwargs.pop("val_hdf5_keys_save_path", None)

        self.shuffle = kwargs.pop("shuffle", None)
        self.train_shuffle = kwargs.pop("train_shuffle", None) or self.shuffle
        self.val_shuffle = kwargs.pop("val_shuffle", None)
        self.test_shuffle = kwargs.pop("test_shuffle", None)

        self.train_data_fraction = kwargs.pop("train_data_fraction", 1.0)
        self.val_data_fraction = kwargs.pop("val_data_fraction", 1.0)
        self.test_data_fraction = kwargs.pop("test_data_fraction", 1.0)

        if self.train_data_fraction != 1.0 and not train_hdf5_keys_path:
            raise ValueError("train_data_fraction provided as non-unity but train_hdf5_keys_path is unset.")
        if self.val_data_fraction != 1.0 and not val_hdf5_keys_path:
            raise ValueError("val_data_fraction provided as non-unity but val_hdf5_keys_path is unset.")
        if self.test_data_fraction != 1.0 and not test_hdf5_keys_path:
            raise ValueError("test_data_fraction provided as non-unity but test_hdf5_keys_path is unset.")

        all_hdf5_data_path = kwargs.pop("all_hdf5_data_path", None)
        if all_hdf5_data_path is not None:
            print(
                "all_hdf5_data_path provided, will be interpreted as the general data path for all splits.\nKeys in provided train_hdf5_keys_path assumed to encompass all keys for entire data. Validation and Test keys will be subtracted from Train keys."
            )
            if self.train_hdf5_path:
                raise ValueError(
                    "Both general all_hdf5_data_path provided and a specific train_hdf5_path, remove the train_hdf5_path"
                )
            if self.val_hdf5_path:
                raise ValueError(
                    "Both general all_hdf5_data_path provided and a specific val_hdf5_path, remove the val_hdf5_path"
                )
            if self.test_hdf5_path:
                raise ValueError(
                    "Both general all_hdf5_data_path provided and a specific test_hdf5_path, remove the test_hdf5_path"
                )
            self.train_hdf5_path = all_hdf5_data_path
            self.val_hdf5_path = all_hdf5_data_path
            self.test_hdf5_path = all_hdf5_data_path
            self.reduce_train_keys = True
        else:
            self.reduce_train_keys = False

        self.resize = kwargs.pop("resize", False)
        self.resize_to = kwargs.pop("resize_to", None)
        if self.resize and self.resize_to is None:
            raise ValueError("Config provided resize as True, but resize_to parameter not given")
        self.resize_interpolation = kwargs.pop("resize_interpolation", None)
        if self.resize and self.resize_interpolation is None:
            print(
                "Config provided resize as True, but resize_interpolation mode not given. Will assume default bilinear"
            )
            self.resize_interpolation = "bilinear"
        interpolation_dict = {
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "nearest": InterpolationMode.NEAREST,
            "nearest_exact": InterpolationMode.NEAREST_EXACT,
        }
        if self.resize:
            if self.resize_interpolation not in interpolation_dict.keys():
                raise ValueError(
                    f"resize_interpolation provided as {self.resize_interpolation}, but valid options are: {interpolation_dict.keys()}"
                )
            self.resize_interpolation = interpolation_dict[self.resize_interpolation]
        self.resize_antialiasing = kwargs.pop("resize_antialiasing", True)

        self.kwargs = kwargs

    def _load_hdf5_keys_from_path(self, path, fraction=1.0):
        if path is None:
            return None
        with open(path, "rb") as f:
            keys = _KeysUnpickler(f).load()
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError(f"HDF5 keys file {path} must contain a list of strings.")
        return keys[: int(fraction * len(keys))]

    def setup(self, stage: str):
        """Set up datasets.

        Args:
            stage: Either fit, test.
        """
        if stage == "fit":
            train_keys = self._load_hdf5_keys_from_path(self.train_hdf5_keys_path, fraction=self.train_data_fraction)
            val_keys = self._load_hdf5_keys_from_path(self.val_hdf5_keys_path, fraction=self.val_data_fraction)
            if self.reduce_train_keys:
                test_keys = self._load_hdf5_keys_from_path(self.test_hdf5_keys_path, fraction=self.test_data_fraction)
                train_keys = list(set(train_keys) - set(val_keys) - set(test_keys))
            train_file = h5py.File(self.train_hdf5_path, "r")
            self.lucasS2_train = Sen4MapDatasetMonthlyComposites(
                train_file,
                h5data_keys=train_keys,
                resize=self.resize,
                resize_to=self.resize_to,
                resize_interpolation=self.resize_interpolation,
                resize_antialiasing=self.resize_antialiasing,
                save_keys_path=self.train_hdf5_keys_save_path,
                **self.kwargs,
            )
            val_file = h5py.File(self.val_hdf5_path, "r")
            self.lucasS2_val = Sen4MapDatasetMonthlyComposites(
                val_file,
                h5data_keys=val_keys,
                resize=self.resize,
                resize_to=self.resize_to,
                resize_interpolation=self.resize_interpolation,
                resize_antialiasing=self.resize_antialiasing,
                save_keys_path=self.val_hdf5_keys_save_path,
                **self.kwargs,
            )
        if stage == "test":
            test_file = h5py.File(self.test_hdf5_path, "r")
            test_keys = self._load_hdf5_keys_from_path(self.test_hdf5_keys_path, fraction=self.test_data_fraction)
            self.lucasS2_test = Sen4MapDatasetMonthlyComposites(
                test_file,
                h5data_keys=test_keys,
                resize=self.resize,
                resize_to=self.resize_to,
                resize_interpolation=self.resize_interpolation,
                resize_antialiasing=self.resize_antialiasing,
                save_keys_path=self.test_hdf5_keys_save_path,
                **self.kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.lucasS2_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=self.train_shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.lucasS2_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=self.val_shuffle,
        )

    def test_dataloader(self):
        return DataLoader(
            self.lucasS2_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=self.test_shuffle,
        )
