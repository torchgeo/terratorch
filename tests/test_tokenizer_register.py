"""Tests for the TerraMind tokenizer registry functions."""

from unittest.mock import patch

import pytest

from terratorch.models.backbones.terramind.tokenizer import tokenizer_register

DIVAE_TOKENIZERS = [
    "terramind_v1_tokenizer_s2l2a",
    "terramind_v1_tokenizer_s1rtc",
    "terramind_v1_tokenizer_s1grd",
    "terramind_v1_tokenizer_dem",
    "terramind_v1_tokenizer_ndvi",
    "terramind_v01_tokenizer_s2l2a",
    "terramind_v01_tokenizer_s1grd",
    "terramind_v01_tokenizer_dem",
]


@pytest.mark.parametrize("name", DIVAE_TOKENIZERS)
def test_divae_tokenizer_clip_sample_default(name):
    with patch.object(tokenizer_register, "build_vqvae") as build_vqvae:
        getattr(tokenizer_register, name)()
    assert build_vqvae.call_args.kwargs["clip_sample"] is True


@pytest.mark.parametrize("name", DIVAE_TOKENIZERS)
def test_divae_tokenizer_clip_sample_override(name):
    with patch.object(tokenizer_register, "build_vqvae") as build_vqvae:
        getattr(tokenizer_register, name)(clip_sample=False, thresholding=False)
    assert build_vqvae.call_args.kwargs["clip_sample"] is False
    assert build_vqvae.call_args.kwargs["thresholding"] is False
