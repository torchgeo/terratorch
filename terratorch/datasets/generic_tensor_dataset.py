import os
import torch
from torch.utils.data import Dataset

class GenericTensorDataset(Dataset):
    def __init__(
        self,
        paths,
        labels_path=None,
        key=None,
        labels_key=None,
        dtype=torch.float32,
        normalize=False,
    ):
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, (list, tuple)):
            raise TypeError("paths must be a string or list of strings")

        tensors = []
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Not a file: {path}")

            obj = torch.load(path, map_location="cpu")

            if isinstance(obj, dict):
                if key is None:
                    raise ValueError(f"{path} contains a dict but no key was provided")
                obj = obj[key]

            if not isinstance(obj, torch.Tensor):
                raise TypeError(f"{path} does not contain a Tensor")

            tensors.append(obj.to(dtype))

        self.tensors = tensors
        self.labels = None
        
        if labels_path is not None:
            if not os.path.isfile(labels_path):
                raise FileNotFoundError(f"Not a file: {labels_path}")

            labels_obj = torch.load(labels_path, map_location="cpu")

            if isinstance(labels_obj, dict):
                if labels_key is None:
                    raise ValueError(
                        f"{labels_path} contains a dict but no labels_key was provided"
                    )
                labels_obj = labels_obj[labels_key]

            if not isinstance(labels_obj, torch.Tensor):
                raise TypeError(f"{labels_path} does not contain a Tensor")

            self.labels = labels_obj

        lengths = [t.shape[0] for t in tensors]
        if self.labels is not None:
            lengths.append(self.labels.shape[0])

        self.length = min(lengths)
        self.normalize = normalize

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        xs = [t[idx] for t in self.tensors]

        if self.normalize:
            xs = [torch.nn.functional.normalize(x, dim=-1) for x in xs]

        batch = {}

        if len(xs) == 1:
            batch["image"] = xs[0]
        else:
            for i, x in enumerate(xs):
                batch[f"image_{i}"] = x

        if self.labels is not None:
            batch["label"] = self.labels[idx].argmax(dim=0) # CHANGE ISA: One hot encode labels

        return batch