import logging

import torch
from torch import nn
from torchgeo.tasks import BaseTask

logger = logging.getLogger(__name__)

from terratorch.registry import MODEL_FACTORY_REGISTRY


class WxCTask(BaseTask):
    def __init__(self, model_factory, model_args: dict, mode: str = "train", learning_rate=0.1):
        if mode not in ["train", "eval"]:
            raise ValueError(f"mode {mode} is not supported. (train, eval)")
        self.model_args = model_args

        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)

        self.learning_rate = learning_rate
        super().__init__()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_models(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model_factory.build_model(**self.model_args)
        self.model = self.model.to(device)
        layer_devices = []
        for name, module in self.model.named_children():
            device = next(module.parameters(), torch.tensor([])).device
            layer_devices.append((name, str(device)))
        logging.debug(layer_devices)

    def training_step(self, batch, batch_idx):
        output: torch.Tensor = self.model(batch)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
