# Copyright contributors to the Terratorch project
"""Tests for terratorch.tasks.loss_handler.

Regression coverage for https://github.com/IBM/terratorch/issues/982: logging the individual
component losses of a CombinedLoss must not force ``on_step``/``on_epoch``. Doing so made DDP
deadlock at the end of the first epoch, because the extra epoch-level synced reduction fired only
for the component losses (and not for the total loss), desynchronising the collective calls across
ranks.
"""
import torch

from terratorch.tasks.loss_handler import CombinedLoss, LossHandler


def _capture_logs(loss_dict, prefix="train/"):
    """Run log_loss with a fake log_function and return the recorded calls."""
    calls = []

    def fake_log(name, value, **kwargs):
        calls.append({"name": name, "value": value, "kwargs": kwargs})

    handler = LossHandler(prefix)
    handler.log_loss(fake_log, loss_dict=loss_dict, batch_size=4)
    return calls


def test_single_loss_logs_only_total():
    calls = _capture_logs({"loss": torch.tensor(1.0)})
    assert [c["name"] for c in calls] == ["train/loss"]
    # No on_step/on_epoch forced; Lightning defaults are used.
    assert "on_step" not in calls[0]["kwargs"]
    assert "on_epoch" not in calls[0]["kwargs"]
    assert calls[0]["kwargs"]["sync_dist"] is True


def test_combined_loss_component_logging_does_not_force_step_epoch():
    loss_dict = {
        "loss": torch.tensor(6.0),
        "ce": torch.tensor(1.0),
        "dice": torch.tensor(5.0),
    }
    calls = _capture_logs(loss_dict)

    names = [c["name"] for c in calls]
    assert names == ["train/loss", "train/ce", "train/dice"]

    # The core of the #982 regression guard: no component (nor the total) may force on_step /
    # on_epoch, so DDP collectives stay symmetric across all logged values.
    for c in calls:
        assert "on_step" not in c["kwargs"], f"{c['name']} must not force on_step"
        assert "on_epoch" not in c["kwargs"], f"{c['name']} must not force on_epoch"
        assert c["kwargs"]["sync_dist"] is True
        assert c["kwargs"]["batch_size"] == 4


def test_log_loss_does_not_mutate_input_dict():
    loss_dict = {"loss": torch.tensor(2.0), "ce": torch.tensor(1.0)}
    _capture_logs(loss_dict)
    assert set(loss_dict.keys()) == {"loss", "ce"}


def test_combined_loss_forward_produces_component_and_total_keys():
    """CombinedLoss returns each named component plus an aggregated "loss" key, and those keys
    are exactly what log_loss emits (total first, then one entry per component)."""
    combined = CombinedLoss({"ce": torch.nn.CrossEntropyLoss(), "ce2": torch.nn.CrossEntropyLoss()})
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))

    out = combined(logits, target)
    assert set(out.keys()) == {"loss", "ce", "ce2"}
    # Unweighted CombinedLoss sums the components.
    torch.testing.assert_close(out["loss"], out["ce"] + out["ce2"])

    calls = _capture_logs(out)
    assert [c["name"] for c in calls] == ["train/loss", "train/ce", "train/ce2"]
