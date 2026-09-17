"""Shared pytest fixtures and guards for the terratorch test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_float32_matmul_precision():
    """Keep ``torch.set_float32_matmul_precision`` from leaking between tests.

    The setting is process-global. CI runs the whole suite in a single process,
    so a module that lowers it at import time silently changes numerics for
    every test that runs afterwards, producing order-dependent failures in
    tolerance-sensitive tests. Restore it around every test.
    """
    previous = torch.get_float32_matmul_precision()
    yield
    if torch.get_float32_matmul_precision() != previous:
        torch.set_float32_matmul_precision(previous)
