from collections.abc import Callable

import torch
from torch import Tensor, nn

from terratorch.models.model import ModelOutput


class LossHandler:
    """Class to help handle the computation and logging of loss"""

    def __init__(self, loss_prefix: str) -> None:
        """Constructor

        Args:
            loss_prefix (str): Prefix to be prepended to all the metrics (e.g. training).
        """
        self.loss_prefix = loss_prefix

    def compute_loss(
        self,
        model_output: ModelOutput,
        ground_truth: Tensor,
        criterion: Callable,
        aux_loss_weights: dict[str, float] | None,
    ) -> dict[str, Tensor]:
        """Compute the loss for the mean decode head as well as other heads

        Args:
            model_output (ModelOutput): Output from the model
            ground_truth (Tensor): Tensor with labels
            criterion (Callable): Loss function to be applied
            aux_loss_weights (Union[dict[str, float], None]): Dictionary of names of model auxiliary
                heads and their weights

        Raises:
            Exception: If the keys in aux_loss_weights and the model output do not match, will raise an exception.

        Returns:
            dict[str, Tensor]: Dictionary of computed losses. Total loss is returned under the key "loss".
                If there are auxiliary heads, the main decode head is returned under the key "decode_head".
                All other heads are returned with the same key as their name.
        """

        loss = self._compute_loss(model_output.output, ground_truth, criterion)
        if isinstance(loss, Tensor):
            loss = {"loss": loss}
        if not model_output.auxiliary_heads:
            return loss

        if aux_loss_weights is None:
            msg = "Auxiliary heads given with no aux_loss_weights"
            raise Exception(msg)
        total_loss = loss["loss"].clone()
        # incorporate aux heads
        model_output_names = set(model_output.auxiliary_heads.keys())
        aux_loss_names = set(aux_loss_weights.keys())
        if aux_loss_names != model_output_names:
            msg = f"Found difference in declared auxiliary losses and model outputs.\n \
                Found in declared losses but not in model output: {aux_loss_names - model_output_names}. \n \
                Found in model output but not in delcared losses: {model_output_names - aux_loss_names}"
            raise Exception(msg)

        for loss_name, loss_weight in aux_loss_weights.items():
            output = model_output.auxiliary_heads[loss_name]
            loss_value: Tensor = self._compute_loss(output, ground_truth, criterion)
            loss[loss_name] = loss_value
            if isinstance(loss_value, dict):
                # Combined loss
                total_loss += loss_value["loss"] * loss_weight
            else:
                total_loss += loss_value * loss_weight

        loss["loss"] = total_loss
        return loss

    def _compute_loss(self, y_hat: Tensor, ground_truth: Tensor, criterion: Callable):
        loss: Tensor = criterion(y_hat, ground_truth)
        return loss

    def log_loss(
        self, log_function: Callable, loss_dict: dict[str, Tensor] | None = None, batch_size: int | None = None
    ) -> None:
        """Log the loss. If auxiliary heads exist, log the full loss suffix "loss", and then all other losses.

        Args:
            log_function (Callable): _description_
            loss_dict (dict[str, Tensor], optional): _description_. Defaults to None.
        """

        # dont alter passed dict
        all_losses = dict(loss_dict)
        full_loss = all_losses.pop("loss")
        log_function(f"{self.loss_prefix}loss", full_loss, sync_dist=True, batch_size=batch_size)

        # Log the individual component losses (present when using a CombinedLoss) with the same
        # settings as the total loss above, i.e. letting Lightning pick on_step/on_epoch from the
        # hook context (on_step during training, on_epoch during validation/test). Forcing both
        # on_step=True and on_epoch=True here previously caused DDP to deadlock at the end of the
        # first epoch, because the extra epoch-level synced reduction was only triggered for the
        # component losses and not for the total loss, desynchronising the collective calls across
        # ranks. See https://github.com/IBM/terratorch/issues/982.
        for loss_name, loss_value in all_losses.items():
            log_function(
                f"{self.loss_prefix}{loss_name}",
                loss_value,
                sync_dist=True,
                batch_size=batch_size,
            )


class CombinedLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, losses: dict[str, nn.Module], weight: list[float] | None = None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        if weight is not None:
            self.weight = torch.Tensor(weight)
        else:
            self.weight = None

    def forward(self, y_hat: Tensor, ground_truth: Tensor) -> dict:
        loss = {name: criterion(y_hat, ground_truth) for name, criterion in self.losses.items()}

        if self.weight is not None:
            # Apply provided loss weights
            loss["loss"] = (torch.stack(list(loss.values())) * self.weight).sum()
        else:
            # Sum up all losses without weighting
            loss["loss"] = torch.stack(list(loss.values())).sum()

        return loss
