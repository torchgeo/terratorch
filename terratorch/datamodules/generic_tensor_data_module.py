from lightning.pytorch import LightningDataModule
from terratorch.datasets.generic_tensor_dataset import GenericTensorDataset
from torch.utils.data import DataLoader



class GenericTensorDataModule(LightningDataModule):
    def __init__(
        self,
        train_paths,
        val_paths=None,
        test_paths=None,
        train_labels_path=None,
        val_labels_path=None,
        test_labels_path=None,
        batch_size=32,
        num_workers=4,
        key=None,
        labels_key=None,
        normalize=False,
        pin_memory=True,
    ):
        super().__init__()

        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths

        self.train_labels_path = train_labels_path
        self.val_labels_path = val_labels_path
        self.test_labels_path = test_labels_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.key = key
        self.labels_key = labels_key
        self.normalize = normalize
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = GenericTensorDataset(
                self.train_paths,
                labels_path=self.train_labels_path,
                key=self.key,
                labels_key=self.labels_key,
                normalize=self.normalize,
            )

            if self.val_paths is not None:
                self.val_dataset = GenericTensorDataset(
                    self.val_paths,
                    labels_path=self.val_labels_path,
                    key=self.key,
                    labels_key=self.labels_key,
                    normalize=self.normalize,
                )

        if stage in (None, "test") and self.test_paths is not None:
            self.test_dataset = GenericTensorDataset(
                self.test_paths,
                labels_path=self.test_labels_path,
                key=self.key,
                labels_key=self.labels_key,
                normalize=self.normalize,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return []
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
