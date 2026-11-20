import math
from collections import Counter
from typing import Dict, Union
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ALLOWED_URL_SCHEMES = ("http", "https")


def check_url_scheme(url: str) -> str:
    """Reject non-HTTP(S) URLs (e.g. ``file://``) before handing them to urllib."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"Unsupported URL scheme {scheme!r}; only http and https are allowed")
    return url


def parse_hf_url(url: str) -> tuple[str, str | None, str]:
    """Split a Hugging Face file URL into ``(repo_id, revision, filename)``.

    Hugging Face file URLs look like ``https://hf.co/<repo_id>/resolve/<revision>/<path/to/file>``.
    ``filename`` keeps the full path inside the repository, so files in subdirectories work.
    If only one segment follows ``/resolve/``, it is taken as the filename and ``revision``
    is ``None`` (the default branch is used).
    """
    repo_id, _, rest = url.replace("https://hf.co/", "", 1).partition("/resolve/")
    parts = [p for p in rest.split("/") if p]
    if len(parts) > 1:
        return repo_id, unquote(parts[0]), "/".join(parts[1:])
    return repo_id, None, parts[0] if parts else ""


# Special Exception to handle cases in which the model is missing from the
# registries
class InvalidModelError(Exception):
    def __init__(self, message, model_name=None):
        if not message:
            message = f"InvalidModelError: The model {model_name} isn't in the register"
        super().__init__(message)


def compute_statistics(dataloader: DataLoader) -> dict[str, list[float]]:
    n_bands = dataloader.dataset[0]["image"].shape[0]
    n_data = torch.zeros([n_bands], dtype=torch.int64)
    sum_data = torch.zeros([n_bands], dtype=torch.float64)

    # First pass for mean
    for batch in tqdm(dataloader, desc="Compute mean"):
        imgs: torch.Tensor = batch["image"]
        # switch batch and band dimensions and flatten
        samples = imgs.transpose(0, 1).reshape(n_bands, -1).double()
        sum_data += samples.nansum(dim=1)
        n_data += samples.shape[1] - samples.isnan().sum(dim=1)
    mean = sum_data / n_data

    sum_squared = torch.zeros(n_bands, dtype=torch.float64)
    for batch in tqdm(dataloader, desc="Compute variance"):
        imgs = batch["image"]
        samples = imgs.transpose(0, 1).reshape(n_bands, -1).double()
        sum_squared += ((samples - mean.unsqueeze(1)) ** 2).nansum(dim=1)

    variance = sum_squared / n_data
    std = torch.sqrt(variance)
    return {"means": mean.numpy().tolist(), "stds": std.numpy().tolist()}


def compute_mask_statistics(dataloader: DataLoader) -> dict[int, dict[str, int | float]] | dict[str, float]:
    if torch.is_floating_point(dataloader.dataset[0]["mask"]):
        return compute_float_mask_statistics(dataloader)
    else:
        return compute_int_mask_statistics(dataloader)


def compute_int_mask_statistics(dataloader: DataLoader) -> dict[int, dict[str, int | float]]:
    counter = Counter()
    for batch in tqdm(dataloader, desc="Compute counts"):
        masks: torch.Tensor = batch["mask"]
        counter.update(masks.flatten().tolist())

    stats = {}
    for key, count in counter.items():
        stats[key] = {"count": count, "percentage": count / counter.total()}
    return stats


def compute_float_mask_statistics(dataloader: DataLoader) -> dict[str, float]:
    n_data = 0
    total = 0.0

    for batch in tqdm(dataloader, desc="Compute mask mean"):
        masks: torch.Tensor = batch["mask"]
        total += masks.sum().item()
        n_data += masks.numel()
    mean = total / n_data

    sum_squared = 0.0
    for batch in tqdm(dataloader, desc="Compute mask variance"):
        masks = batch["mask"]
        sum_squared += ((masks - mean) ** 2).sum().item()

    variance = sum_squared / n_data
    std = math.sqrt(variance)
    return {"mean": mean, "std": std}


# TODO remove it for future releases
def remove_unexpected_prefix(state_dict):
    state_dict_ = {}
    for k, v in state_dict.items():
        keys = k.split(".")
        if "_timm_module" in keys:
            index = keys.index("_timm_module")
            keys.pop(index)
            k_ = ".".join(keys)
        else:
            k_ = k
        state_dict_[k_] = v
    return state_dict_


def view_api(
    module: torch.nn.Module = None,
    input_data: torch.nn.Module | np.ndarray = None,
    config: dict = None,
    save_dir: str = None,
):
    try:
        from torchview import draw_graph
    except Exception:
        raise Exception(
            "For using the visualization API,\
                        it is necessary to install torchview and graphviz"
        )

    # Wrapper for torchview
    class Module_wrap(torch.nn.Module):
        def __init__(self, module=None):
            super(Module_wrap, self).__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args)

        def train(self, *args, **kwargs):
            pass

        def eval(self, *args, **kwargs):
            pass

    module_wrap = Module_wrap(module=module)

    model_graph = draw_graph(module_wrap, input_data=input_data, directory=save_dir, **config)

    return model_graph.visual_graph
