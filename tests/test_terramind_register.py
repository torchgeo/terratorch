"""Comprehensive tests for terramind_register.py to maximize coverage."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import warnings
from unittest.mock import patch, MagicMock
from functools import partial

from terratorch.models.backbones.terramind.model.terramind_register import (
    select_modality_patch_embed_weights,
    checkpoint_filter_fn,
    checkpoint_filter_fn_tim,
    checkpoint_filter_fn_generate,
    build_terrammind_vit,
    build_terrammind_encdec,
    build_terrammind_tim,
    build_terrammind_generate,
    PRETRAINED_BANDS,
    v01_pretraining_mean,
    v01_pretraining_std,
    v1_pretraining_mean,
    v1_pretraining_std,
    tokenizer_dict,
    pretrained_weights,
)
from terratorch.models.backbones.terramind.model.terramind_vit import TerraMindViT
from terratorch.models.backbones.terramind.model.terramind import TerraMind
from terratorch.models.backbones.terramind.model.terramind_tim import TerraMindTiM
from terratorch.models.backbones.terramind.model.terramind_generation import TerraMindGeneration


class MockPatchEmbed(nn.Module):
    """Mock patch embedding for testing"""
    def __init__(self, dim_tokens=64, patch_size=(16, 16)):
        super().__init__()
        self.dim_tokens = dim_tokens
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size[0] * patch_size[1] * 3, dim_tokens, bias=False)
        
    def forward(self, x):
        return x


def create_mock_vit_model(modalities=['mod1', 'mod2']):
    """Create a minimal mock TerraMindViT model for testing"""
    model = MagicMock(spec=TerraMindViT)
    model.mod_name_mapping = {mod: mod for mod in modalities}
    model.encoder_embeddings = {}
    for mod in modalities:
        mock_emb = MagicMock()
        mock_emb.patch_size = (16, 16)
        mock_emb.dim_tokens = 64
        mock_emb.proj = nn.Linear(16*16*3, 64, bias=False)
        model.encoder_embeddings[mod] = mock_emb
    
    # Mock state_dict
    state_dict = {}
    for mod in modalities:
        state_dict[f'encoder_embeddings.{mod}.proj.weight'] = torch.randn(64, 16*16*3)
    model.state_dict.return_value = state_dict
    
    return model


# Test select_modality_patch_embed_weights
def test_select_modality_patch_embed_weights_basic():
    """Test basic band weight selection"""
    model = create_mock_vit_model(['untok_sen2l2a@224'])
    # Use only bands that exist in pretrained
    bands = {'untok_sen2l2a@224': ['RED', 'GREEN', 'BLUE']}
    pretrained_bands = {'untok_sen2l2a@224': ['COASTAL_AEROSOL', 'BLUE', 'GREEN', 'RED']}
    
    # Update mock to have correct input dimensions for 4 bands (pretrained)
    model.encoder_embeddings['untok_sen2l2a@224'].proj = nn.Linear(16*16*4, 64, bias=False)
    
    result = select_modality_patch_embed_weights(model, bands, pretrained_bands)
    # Check that new proj layer has correct input dimension (3 bands)
    assert model.encoder_embeddings['untok_sen2l2a@224'].proj.weight.shape[1] == 16*16*3


def test_select_modality_patch_embed_weights_missing_modality():
    """Test when modality not in pretrained bands"""
    with patch('terratorch.models.backbones.terramind.model.terramind_register.logger') as mock_logger:
        model = create_mock_vit_model(['custom_mod'])
        bands = {'custom_mod': ['BAND1']}
        pretrained_bands = {}
        
        result = select_modality_patch_embed_weights(model, bands, pretrained_bands)
        # Check logger.info was called with appropriate message
        assert any('Cannot load band weights' in str(call) for call in mock_logger.info.call_args_list)


def test_select_modality_patch_embed_weights_band_subset():
    """Test selecting subset of bands"""
    model = create_mock_vit_model(['untok_sen2rgb@224'])
    bands = {'untok_sen2rgb@224': ['RED', 'GREEN']}
    pretrained_bands = {'untok_sen2rgb@224': ['RED', 'GREEN', 'BLUE']}
    
    result = select_modality_patch_embed_weights(model, bands, pretrained_bands)
    # Check that new proj layer has correct input dimension (2 bands)
    assert model.encoder_embeddings['untok_sen2rgb@224'].proj.weight.shape[1] == 16*16*2


# Test checkpoint_filter_fn
def test_checkpoint_filter_fn_matching_shapes():
    """Test checkpoint filter with matching shapes"""
    model = MagicMock(spec=TerraMindViT)
    model.state_dict.return_value = {
        'encoder.weight': torch.randn(10, 20),
        'decoder.weight': torch.randn(5, 10)
    }
    
    state_dict = {
        'encoder.weight': torch.randn(10, 20),
        'decoder.weight': torch.randn(5, 10)
    }
    
    result = checkpoint_filter_fn(state_dict, model)
    assert 'encoder.weight' in result
    assert 'decoder.weight' in result


def test_checkpoint_filter_fn_shape_mismatch(caplog):
    """Test checkpoint filter with shape mismatch"""
    model = MagicMock(spec=TerraMindViT)
    model.state_dict.return_value = {
        'encoder.weight': torch.randn(10, 20),
    }
    
    state_dict = {
        'encoder.weight': torch.randn(10, 30),  # Different shape
    }
    
    result = checkpoint_filter_fn(state_dict, model)
    assert "Shape for encoder.weight" in caplog.text
    assert result['encoder.weight'].shape == (10, 20)  # Should use model weights


def test_checkpoint_filter_fn_missing_weights(caplog):
    """Test checkpoint filter with missing weights"""
    model = MagicMock(spec=TerraMindViT)
    model.state_dict.return_value = {
        'encoder.weight': torch.randn(10, 20),
        'new_layer.weight': torch.randn(5, 5),
    }
    
    state_dict = {
        'encoder.weight': torch.randn(10, 20),
    }
    
    result = checkpoint_filter_fn(state_dict, model)
    assert "Weights for new_layer.weight are missing" in caplog.text


def test_checkpoint_filter_fn_tokenizer_no_warning(caplog):
    """Test that tokenizer weights don't trigger warnings"""
    model = MagicMock(spec=TerraMindViT)
    model.state_dict.return_value = {
        'encoder.weight': torch.randn(10, 20),
        'tokenizer.weight': torch.randn(5, 5),
    }
    
    state_dict = {
        'encoder.weight': torch.randn(10, 20),
    }
    
    result = checkpoint_filter_fn(state_dict, model)
    # Check no warning for tokenizer
    assert "tokenizer" not in caplog.text or "missing" not in caplog.text.lower()


# Test checkpoint_filter_fn_tim
def test_checkpoint_filter_fn_tim_with_sampler():
    """Test TiM checkpoint filter with sampler.model prefix"""
    model = MagicMock(spec=TerraMindTiM)
    model.state_dict.return_value = {
        'encoder.weight': torch.randn(10, 20),
        'sampler.model.encoder.weight': torch.randn(10, 20),
    }
    
    state_dict = {
        'encoder.weight': torch.randn(10, 20),
    }
    
    result = checkpoint_filter_fn_tim(state_dict, model)
    assert 'encoder.weight' in result
    assert 'sampler.model.encoder.weight' in result


def test_checkpoint_filter_fn_tim_shape_mismatch_raises():
    """Test TiM filter raises error on sampler shape mismatch"""
    model = MagicMock(spec=TerraMindTiM)
    model.state_dict.return_value = {
        'sampler.model.encoder.weight': torch.randn(10, 20),
    }
    
    state_dict = {
        'encoder.weight': torch.randn(10, 30),  # Wrong shape
    }
    
    with pytest.raises(ValueError, match="Cannot run chain of thoughts without MAE"):
        checkpoint_filter_fn_tim(state_dict, model)


def test_checkpoint_filter_fn_tim_missing_sampler_raises():
    """Test TiM filter raises error when sampler weights missing"""
    model = MagicMock(spec=TerraMindTiM)
    model.state_dict.return_value = {
        'encoder.weight': torch.randn(10, 20),
        'sampler.model.decoder.weight': torch.randn(5, 10),
    }
    
    state_dict = {
        'encoder.weight': torch.randn(10, 20),
    }
    
    with pytest.raises(ValueError, match="cannot run chain of thoughts without MAE"):
        checkpoint_filter_fn_tim(state_dict, model)


# Test checkpoint_filter_fn_generate
def test_checkpoint_filter_fn_generate_basic():
    """Test generate checkpoint filter"""
    model = MagicMock(spec=TerraMindGeneration)
    model.state_dict.return_value = {
        'sampler.model.encoder.weight': torch.randn(10, 20),
        'other.weight': torch.randn(5, 5),
    }
    
    state_dict = {
        'encoder.weight': torch.randn(10, 20),
    }
    
    result = checkpoint_filter_fn_generate(state_dict, model)
    assert 'sampler.model.encoder.weight' in result


def test_checkpoint_filter_fn_generate_tokenizer_no_warning(caplog):
    """Test generate filter doesn't warn about tokenizer"""
    model = MagicMock(spec=TerraMindGeneration)
    model.state_dict.return_value = {
        'sampler.model.encoder.weight': torch.randn(10, 20),
        'tokenizer.weight': torch.randn(5, 5),
    }
    
    state_dict = {}
    
    result = checkpoint_filter_fn_generate(state_dict, model)
    # Should not warn about tokenizer
    log_text = caplog.text.lower()
    assert 'tokenizer' not in log_text or 'missing' not in log_text


# Test build_terrammind_vit with various scenarios
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindViT')
def test_build_terrammind_vit_with_ckpt_path(mock_vit):
    """Test building ViT with checkpoint path"""
    mock_model = MagicMock()
    mock_model.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
    mock_vit.return_value = mock_model
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save({'encoder.weight': torch.randn(10, 20)}, tmp.name)
        tmp_path = tmp.name
    
    try:
        model = build_terrammind_vit(ckpt_path=tmp_path, pretrained=False)
        mock_model.load_state_dict.assert_called_once()
    finally:
        os.unlink(tmp_path)


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindViT')
def test_build_terrammind_vit_with_missing_keys(mock_vit, caplog):
    """Test building ViT with missing keys warning"""
    mock_model = MagicMock()
    mock_model.load_state_dict.return_value = MagicMock(
        missing_keys=['layer.weight'], 
        unexpected_keys=['extra.weight']
    )
    mock_vit.return_value = mock_model
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save({'encoder.weight': torch.randn(10, 20)}, tmp.name)
        tmp_path = tmp.name
    
    try:
        model = build_terrammind_vit(ckpt_path=tmp_path, pretrained=False)
        assert "Missing keys" in caplog.text
    finally:
        os.unlink(tmp_path)


@patch('terratorch.models.backbones.terramind.model.terramind_register.hf_hub_download')
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindViT')
@patch('terratorch.models.backbones.terramind.model.terramind_register.checkpoint_filter_fn')
def test_build_terrammind_vit_pretrained(mock_filter, mock_vit, mock_hub):
    """Test building pretrained ViT from HuggingFace"""
    mock_model = MagicMock()
    mock_vit.return_value = mock_model
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save({'encoder.weight': torch.randn(10, 20)}, tmp.name)
        tmp_path = tmp.name
        mock_hub.return_value = tmp_path
    
    try:
        mock_filter.return_value = {'encoder.weight': torch.randn(10, 20)}
        model = build_terrammind_vit(variant='terramind_v1_base', pretrained=True)
        mock_hub.assert_called_once()
        mock_filter.assert_called_once()
    finally:
        os.unlink(tmp_path)


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindViT')
@patch('terratorch.models.backbones.terramind.model.terramind_register.select_modality_patch_embed_weights')
def test_build_terrammind_vit_with_bands(mock_select, mock_vit):
    """Test building ViT with custom bands"""
    mock_model = MagicMock()
    mock_vit.return_value = mock_model
    mock_select.return_value = mock_model
    
    bands = {'untok_sen2rgb@224': ['RED', 'GREEN', 'BLUE']}
    model = build_terrammind_vit(
        pretrained=False, 
        bands=bands, 
        pretrained_bands=PRETRAINED_BANDS
    )
    mock_select.assert_called_once_with(mock_model, bands, PRETRAINED_BANDS)


# Test build_terrammind_encdec
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMind')
def test_build_terrammind_encdec_basic(mock_encdec):
    """Test building encoder-decoder model"""
    mock_model = MagicMock()
    mock_encdec.return_value = mock_model
    
    model = build_terrammind_encdec(pretrained=False)
    mock_encdec.assert_called_once()


@patch('terratorch.models.backbones.terramind.model.terramind_register.hf_hub_download')
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMind')
@patch('terratorch.models.backbones.terramind.model.terramind_register.checkpoint_filter_fn')
def test_build_terrammind_encdec_pretrained(mock_filter, mock_encdec, mock_hub):
    """Test building pretrained encoder-decoder"""
    mock_model = MagicMock()
    mock_encdec.return_value = mock_model
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save({'encoder.weight': torch.randn(10, 20)}, tmp.name)
        tmp_path = tmp.name
        mock_hub.return_value = tmp_path
    
    try:
        mock_filter.return_value = {'encoder.weight': torch.randn(10, 20)}
        model = build_terrammind_encdec(variant='terramind_v1_base', pretrained=True)
        mock_hub.assert_called_once()
    finally:
        os.unlink(tmp_path)


# Test build_terrammind_tim
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindTiM')
def test_build_terrammind_tim_with_bands_raises(mock_tim):
    """Test TiM raises error when bands provided"""
    mock_model = MagicMock()
    mock_tim.return_value = mock_model
    
    bands = {'untok_sen2rgb@224': ['RED', 'GREEN', 'BLUE']}
    with pytest.raises(NotImplementedError, match="Bands cannot be adapted"):
        build_terrammind_tim(pretrained=False, bands=bands)


@patch('terratorch.models.backbones.terramind.model.terramind_register.hf_hub_download')
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindTiM')
def test_build_terrammind_tim_new_modalities_raises(mock_tim, mock_hub):
    """Test TiM raises error with new modalities when pretrained"""
    mock_model = MagicMock()
    mock_tim.return_value = mock_model
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save({'encoder.weight': torch.randn(10, 20)}, tmp.name)
        tmp_path = tmp.name
        mock_hub.return_value = tmp_path
    
    try:
        with pytest.raises(NotImplementedError, match="do not support new modalities"):
            build_terrammind_tim(
                variant='terramind_v1_base',
                pretrained=True,
                modalities=[{'name': 'new_mod', 'type': 'img'}]
            )
    finally:
        os.unlink(tmp_path)


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindTiM')
def test_build_terrammind_tim_not_pretrained_warns(mock_tim):
    """Test TiM warns when not pretrained"""
    mock_model = MagicMock()
    mock_tim.return_value = mock_model
    
    with pytest.warns(UserWarning, match="not pre-trained"):
        model = build_terrammind_tim(pretrained=False)


# Test build_terrammind_generate
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindGeneration')
def test_build_terrammind_generate_basic(mock_gen):
    """Test building generation model"""
    mock_model = MagicMock()
    mock_gen.return_value = mock_model
    
    model = build_terrammind_generate(pretrained=False)
    mock_gen.assert_called_once()


@patch('terratorch.models.backbones.terramind.model.terramind_register.hf_hub_download')
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindGeneration')
@patch('terratorch.models.backbones.terramind.model.terramind_register.checkpoint_filter_fn_generate')
def test_build_terrammind_generate_pretrained(mock_filter, mock_gen, mock_hub):
    """Test building pretrained generation model"""
    mock_model = MagicMock()
    mock_gen.return_value = mock_model
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save({'encoder.weight': torch.randn(10, 20)}, tmp.name)
        tmp_path = tmp.name
        mock_hub.return_value = tmp_path
    
    try:
        mock_filter.return_value = {'sampler.model.encoder.weight': torch.randn(10, 20)}
        model = build_terrammind_generate(variant='terramind_v1_base', pretrained=True)
        mock_hub.assert_called_once()
        mock_filter.assert_called_once()
    finally:
        os.unlink(tmp_path)


# Test constants and data structures
def test_pretrained_bands_structure():
    """Test PRETRAINED_BANDS has expected structure"""
    assert 'untok_sen2l2a@224' in PRETRAINED_BANDS
    assert 'untok_sen2rgb@224' in PRETRAINED_BANDS
    assert isinstance(PRETRAINED_BANDS['untok_sen2l2a@224'], list)
    assert len(PRETRAINED_BANDS['untok_sen2l2a@224']) == 12


def test_pretraining_mean_std_consistency():
    """Test pretraining mean/std have matching keys"""
    for key in v01_pretraining_mean:
        assert key in v01_pretraining_std, f"Key {key} in mean but not in std"
    
    for key in v1_pretraining_mean:
        assert key in v1_pretraining_std, f"Key {key} in mean but not in std"


def test_tokenizer_dict_structure():
    """Test tokenizer_dict has expected structure"""
    assert 'v1' in tokenizer_dict
    assert 'v01' in tokenizer_dict
    assert 'tok_sen2l2a@224' in tokenizer_dict['v1']
    assert 'coords' in tokenizer_dict['v1']


def test_pretrained_weights_structure():
    """Test pretrained_weights dictionary"""
    assert 'terramind_v1_base' in pretrained_weights
    assert 'hf_hub_id' in pretrained_weights['terramind_v1_base']
    assert 'hf_hub_filename' in pretrained_weights['terramind_v1_base']


# Tests for model registration functions
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindViT')
def test_terramind_v1_tiny(mock_vit_class):
    """Test terramind_v1_tiny registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_tiny
    
    mock_model = MagicMock()
    mock_vit_class.return_value = mock_model
    
    model = terramind_v1_tiny(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_vit_class.called
    call_kwargs = mock_vit_class.call_args[1]
    assert call_kwargs['dim'] == 192
    assert call_kwargs['num_heads'] == 3
    assert call_kwargs['encoder_depth'] == 12


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindViT')
def test_terramind_v1_small(mock_vit_class):
    """Test terramind_v1_small registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_small
    
    mock_model = MagicMock()
    mock_vit_class.return_value = mock_model
    
    model = terramind_v1_small(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_vit_class.called
    call_kwargs = mock_vit_class.call_args[1]
    assert call_kwargs['dim'] == 384
    assert call_kwargs['num_heads'] == 6
    assert call_kwargs['encoder_depth'] == 12


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindViT')
def test_terramind_v1_large(mock_vit_class):
    """Test terramind_v1_large registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_large
    
    mock_model = MagicMock()
    mock_vit_class.return_value = mock_model
    
    model = terramind_v1_large(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_vit_class.called
    call_kwargs = mock_vit_class.call_args[1]
    assert call_kwargs['dim'] == 1024
    assert call_kwargs['num_heads'] == 16
    assert call_kwargs['encoder_depth'] == 24


@patch.dict(os.environ, {}, clear=True)
@patch('terratorch.models.backbones.terramind.model.terramind_register.hf_hub_download')
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindViT')
def test_terramind_v01_base_no_hf_token_warning(mock_vit_class, mock_hub):
    """Test terramind_v01_base warns when pretrained=True but no HF_TOKEN."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v01_base
    
    mock_model = MagicMock()
    mock_vit_class.return_value = mock_model
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save({'encoder.weight': torch.randn(10, 20)}, tmp.name)
        tmp_path = tmp.name
        mock_hub.return_value = tmp_path
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = terramind_v01_base(pretrained=True, modalities=["untok_sen2l2a@224"])
            assert len(w) >= 1
            assert any("HF_TOKEN" in str(warning.message) for warning in w)
    finally:
        os.unlink(tmp_path)


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindTiM')
def test_terramind_v1_tiny_tim(mock_tim_class):
    """Test terramind_v1_tiny_tim registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_tiny_tim
    
    mock_model = MagicMock()
    mock_tim_class.return_value = mock_model
    
    model = terramind_v1_tiny_tim(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_tim_class.called
    call_kwargs = mock_tim_class.call_args[1]
    assert call_kwargs['dim'] == 192
    assert call_kwargs['decoder_depth'] == 4


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindTiM')
def test_terramind_v1_small_tim(mock_tim_class):
    """Test terramind_v1_small_tim registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_small_tim
    
    mock_model = MagicMock()
    mock_tim_class.return_value = mock_model
    
    model = terramind_v1_small_tim(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_tim_class.called
    call_kwargs = mock_tim_class.call_args[1]
    assert call_kwargs['dim'] == 384
    assert call_kwargs['decoder_depth'] == 6


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindTiM')
def test_terramind_v1_large_tim(mock_tim_class):
    """Test terramind_v1_large_tim registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_large_tim
    
    mock_model = MagicMock()
    mock_tim_class.return_value = mock_model
    
    model = terramind_v1_large_tim(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_tim_class.called
    call_kwargs = mock_tim_class.call_args[1]
    assert call_kwargs['dim'] == 1024
    assert call_kwargs['decoder_depth'] == 24


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMind')
def test_terramind_v1_tiny_encdec(mock_encdec_class):
    """Test terramind_v1_tiny_encdec registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_tiny_encdec
    
    mock_model = MagicMock()
    mock_encdec_class.return_value = mock_model
    
    model = terramind_v1_tiny_encdec(
        pretrained=False,
        encoder_embeddings={},
        decoder_embeddings={},
        modality_info={}
    )
    
    assert mock_encdec_class.called
    call_kwargs = mock_encdec_class.call_args[1]
    assert call_kwargs['dim'] == 192


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMind')
def test_terramind_v1_small_encdec(mock_encdec_class):
    """Test terramind_v1_small_encdec registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_small_encdec
    
    mock_model = MagicMock()
    mock_encdec_class.return_value = mock_model
    
    model = terramind_v1_small_encdec(
        pretrained=False,
        encoder_embeddings={},
        decoder_embeddings={},
        modality_info={}
    )
    
    assert mock_encdec_class.called
    call_kwargs = mock_encdec_class.call_args[1]
    assert call_kwargs['dim'] == 384


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindGeneration')
def test_terramind_v1_tiny_generate(mock_gen_class):
    """Test terramind_v1_tiny_generate registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_tiny_generate
    
    mock_model = MagicMock()
    mock_gen_class.return_value = mock_model
    
    model = terramind_v1_tiny_generate(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_gen_class.called
    call_kwargs = mock_gen_class.call_args[1]
    assert call_kwargs['dim'] == 192
    assert call_kwargs['decoder_depth'] == 4


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindGeneration')
def test_terramind_v1_small_generate(mock_gen_class):
    """Test terramind_v1_small_generate registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_small_generate
    
    mock_model = MagicMock()
    mock_gen_class.return_value = mock_model
    
    model = terramind_v1_small_generate(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_gen_class.called
    call_kwargs = mock_gen_class.call_args[1]
    assert call_kwargs['dim'] == 384
    assert call_kwargs['decoder_depth'] == 6


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindGeneration')
def test_terramind_v1_large_generate(mock_gen_class):
    """Test terramind_v1_large_generate registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_large_generate
    
    mock_model = MagicMock()
    mock_gen_class.return_value = mock_model
    
    model = terramind_v1_large_generate(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_gen_class.called
    call_kwargs = mock_gen_class.call_args[1]
    assert call_kwargs['dim'] == 1024
    assert call_kwargs['decoder_depth'] == 24


@patch.dict(os.environ, {}, clear=True)
@patch('terratorch.models.backbones.terramind.model.terramind_register.hf_hub_download')
@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindGeneration')
def test_terramind_v01_base_generate_no_hf_token_warning(mock_gen_class, mock_hub):
    """Test terramind_v01_base_generate warns when pretrained=True but no HF_TOKEN."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v01_base_generate
    
    mock_model = MagicMock()
    mock_gen_class.return_value = mock_model
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save({'encoder.weight': torch.randn(10, 20)}, tmp.name)
        tmp_path = tmp.name
        mock_hub.return_value = tmp_path
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = terramind_v01_base_generate(pretrained=True, modalities=["untok_sen2l2a@224"])
            assert len(w) >= 1
            assert any("HF_TOKEN" in str(warning.message) for warning in w)
    finally:
        os.unlink(tmp_path)


@patch('terratorch.models.backbones.terramind.model.terramind_register.TerraMindTiM')
def test_terramind_v01_base_tim(mock_tim_class):
    """Test terramind_v01_base_tim registration function."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v01_base_tim
    
    mock_model = MagicMock()
    mock_tim_class.return_value = mock_model
    
    model = terramind_v01_base_tim(pretrained=False, modalities=["untok_sen2l2a@224"])
    
    assert mock_tim_class.called
    call_kwargs = mock_tim_class.call_args[1]
    assert call_kwargs['dim'] == 768
    assert call_kwargs['encoder_depth'] == 12


def test_terramind_v1_base_encdec_missing_args():
    """Test terramind_v1_base_encdec raises AssertionError when missing required kwargs."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_base_encdec
    
    with pytest.raises(AssertionError):
        terramind_v1_base_encdec(pretrained=False)


def test_terramind_v1_large_encdec_missing_args():
    """Test terramind_v1_large_encdec raises AssertionError when missing required kwargs."""
    from terratorch.models.backbones.terramind.model.terramind_register import terramind_v1_large_encdec
    
    with pytest.raises(AssertionError):
        terramind_v1_large_encdec(pretrained=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



def test_terramind_patch_size():
    """patch_size reaches the pre-trained modality embeddings; with pre-trained weights it warns."""
    model = build_terrammind_vit(modalities=["S2L2A"], patch_size=8, encoder_depth=1, dim=64, num_heads=1,
                                 tokenizer_dict=tokenizer_dict["v1"])
    assert model.encoder_embeddings["untok_sen2l2a@224"].patch_size == (8, 8)
    assert model({"S2L2A": torch.randn(1, 12, 32, 32)})[-1].shape == (1, 16, 64)
    with patch("terratorch.models.backbones.terramind.model.terramind_register.hf_hub_download"), \
            patch("torch.load", return_value=model.state_dict()), pytest.warns(UserWarning, match="randomly"):
        build_terrammind_vit(variant="terramind_v1_tiny", pretrained=True, modalities=["S2L2A"], patch_size=8,
                             encoder_depth=1, dim=64, num_heads=1, tokenizer_dict=tokenizer_dict["v1"])
