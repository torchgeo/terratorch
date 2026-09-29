import os

import torch
from granitewxc.models.loss import rmse_loss
from lightning.pytorch import Callback
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchgeo.tasks import BaseTask
from torchmetrics.functional import mean_absolute_error, structural_similarity_index_measure


def log_spectral_distance(predictions, targets):
    """Computes the Log Spectral Distance (LSD) between predictions and targets"""

    pred_fft = torch.fft.rfft2(predictions)
    target_fft = torch.fft.rfft2(targets)

    pred_magnitude = torch.abs(pred_fft)
    target_magnitude = torch.abs(target_fft)

    # avoid log(0) by adding a small epsilon
    epsilon = 1e-10
    lsd = torch.mean(torch.abs(torch.log10(pred_magnitude + epsilon) - torch.log10(target_magnitude + epsilon)))

    return lsd


class ECCCTask(BaseTask):
    def __init__(self, model_factory, model_args):
        if not model_args:
            raise ValueError("model_args must be provided.")

        self.model_factory = model_factory
        self.model_args = model_args
        self.loss_fn = rmse_loss
        self.learning_rate = model_args.learning_rate
        super().__init__()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def configure_models(self):
        self.model = self.model_factory.build_model(
            backbone="prithvi-eccc-downscaling",
            aux_decoders=None,
            model_args=self.model_args,
            checkpoint_path=self.model_args.path_model_weights,
        )

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage):
        prediction = self.model(batch).output
        loss = self.loss_fn(prediction, batch)

        self.log(f"{stage}_loss", loss, on_epoch=True, sync_dist=True)

        target = batch["y"]

        lsd = log_spectral_distance(prediction, target)
        mae = mean_absolute_error(prediction, target)
        mssim = structural_similarity_index_measure(prediction, target)
        self.log(f"{stage}_LSD", lsd, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_MAE", mae, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_MSSIM", mssim, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")


class CheckpointCallback(Callback):
    def __init__(self, config, save_every_n_epochs, save_dir="pretrained"):
        super().__init__()
        self.config = config
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch % self.save_every_n_epochs == 0 and epoch > 0:
            train_loss = trainer.callback_metrics.get("train_loss", None)
            curr_val_loss = trainer.callback_metrics.get("val_loss", None)
            optimizer = trainer.optimizers[0]

            checkpoint_name = f"checkpoint_{epoch}.pt"
            checkpoint_dir = os.path.join(self.config.path_experiment, self.save_dir)
            checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Prepare the state dictionary
            state_dict = {
                "model": pl_module.state_dict(),  # Save only model weights
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": train_loss.item() if train_loss else None,
                "val_loss": curr_val_loss.item() if curr_val_loss else None,
            }

            torch.save(state_dict, checkpoint_file)
            print(f"\n--> Saved checkpoint: {checkpoint_file}")
