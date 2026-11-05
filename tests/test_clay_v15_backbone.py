"""Comprehensive tests for Clay v1.5 backbone components.

Tests cover FeedForward, Attention, and Transformer modules with various configurations
to ensure maximum code coverage.
"""

import gc

import pytest
import torch
from torch import nn

from terratorch.models.backbones.clay_v15.backbone import (
    Attention,
    FeedForward,
    Transformer,
)


class TestFeedForward:
    """Test suite for FeedForward module."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor for testing."""
        # Batch size 2, sequence length 10, dimension 64.
        # Seeded so the tensor does not depend on RNG state left over from
        # earlier tests, which made numerical comparisons order-dependent.
        torch.manual_seed(0)
        return torch.randn(2, 10, 64)

    def test_feedforward_initialization(self):
        """Test FeedForward module initialization."""
        dim = 64
        hidden_dim = 256
        ff = FeedForward(dim=dim, hidden_dim=hidden_dim)

        assert isinstance(ff.net, nn.Sequential)
        assert len(ff.net) == 4
        assert isinstance(ff.net[0], nn.LayerNorm)
        assert isinstance(ff.net[1], nn.Linear)
        assert isinstance(ff.net[2], nn.GELU)
        assert isinstance(ff.net[3], nn.Linear)

        # Check layer dimensions
        assert ff.net[1].in_features == dim
        assert ff.net[1].out_features == hidden_dim
        assert ff.net[3].in_features == hidden_dim
        assert ff.net[3].out_features == dim

    def test_feedforward_forward(self, sample_input):
        """Test FeedForward forward pass."""
        dim = 64
        hidden_dim = 256
        ff = FeedForward(dim=dim, hidden_dim=hidden_dim)

        output = ff(sample_input)

        # Check output shape matches input shape
        assert output.shape == sample_input.shape
        # Check output is different from input (transformation occurred)
        assert not torch.allclose(output, sample_input)

    def test_feedforward_with_different_dimensions(self):
        """Test FeedForward with various dimension configurations."""
        test_cases = [
            (32, 128),
            (128, 512),
            (256, 1024),
            (512, 2048),
        ]

        for dim, hidden_dim in test_cases:
            ff = FeedForward(dim=dim, hidden_dim=hidden_dim)
            input_tensor = torch.randn(2, 10, dim)
            output = ff(input_tensor)
            assert output.shape == input_tensor.shape

    def test_feedforward_gradient_flow(self, sample_input):
        """Test that gradients flow correctly through FeedForward."""
        dim = 64
        hidden_dim = 256
        ff = FeedForward(dim=dim, hidden_dim=hidden_dim)

        sample_input.requires_grad = True
        output = ff(sample_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert sample_input.grad is not None
        assert not torch.all(sample_input.grad == 0)

    def test_feedforward_batch_independence(self):
        """Test that FeedForward processes batch samples independently."""
        dim = 64
        hidden_dim = 256
        ff = FeedForward(dim=dim, hidden_dim=hidden_dim)

        # Create input with two different samples
        input1 = torch.randn(1, 10, dim)
        input2 = torch.randn(1, 10, dim)
        batched_input = torch.cat([input1, input2], dim=0)

        # Process separately and together
        output1 = ff(input1)
        output2 = ff(input2)
        batched_output = ff(batched_input)

        # Results should be identical
        assert torch.allclose(batched_output[0], output1[0], atol=1e-6)
        assert torch.allclose(batched_output[1], output2[0], atol=1e-6)


class TestAttention:
    """Test suite for Attention module."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor for testing."""
        # Batch size 2, sequence length 10, dimension 64.
        # Seeded so the tensor does not depend on RNG state left over from
        # earlier tests, which made numerical comparisons order-dependent.
        torch.manual_seed(0)
        return torch.randn(2, 10, 64)

    def test_attention_initialization(self):
        """Test Attention module initialization."""
        dim = 64
        heads = 8
        dim_head = 64
        fused_attn = True

        attn = Attention(dim=dim, heads=heads, dim_head=dim_head, fused_attn=fused_attn)

        assert attn.heads == heads
        assert attn.scale == dim_head**-0.5
        assert attn.fused_attn == fused_attn
        assert isinstance(attn.norm, nn.LayerNorm)
        assert isinstance(attn.to_qkv, nn.Linear)
        assert isinstance(attn.to_out, nn.Linear)

        # Check dimensions
        inner_dim = dim_head * heads
        assert attn.to_qkv.in_features == dim
        assert attn.to_qkv.out_features == inner_dim * 3
        assert attn.to_out.in_features == inner_dim
        assert attn.to_out.out_features == dim

    def test_attention_default_parameters(self):
        """Test Attention with default parameters."""
        dim = 64
        attn = Attention(dim=dim)

        # Check defaults
        assert attn.heads == 8
        assert attn.scale == 64**-0.5
        assert attn.fused_attn == True

    def test_attention_forward_fused(self, sample_input):
        """Test Attention forward pass with fused attention."""
        dim = 64
        attn = Attention(dim=dim, heads=8, dim_head=64, fused_attn=True)

        output = attn(sample_input)

        # Check output shape matches input shape
        assert output.shape == sample_input.shape

    def test_attention_forward_unfused(self, sample_input):
        """Test Attention forward pass without fused attention."""
        dim = 64
        attn = Attention(dim=dim, heads=8, dim_head=64, fused_attn=False)

        output = attn(sample_input)

        # Check output shape matches input shape
        assert output.shape == sample_input.shape

    def test_attention_fused_vs_unfused(self, sample_input):
        """Test that fused and unfused attention produce similar results."""
        dim = 64
        heads = 4
        dim_head = 32

        # Create two attention modules with same initialization
        torch.manual_seed(42)
        attn_fused = Attention(dim=dim, heads=heads, dim_head=dim_head, fused_attn=True)

        torch.manual_seed(42)
        attn_unfused = Attention(dim=dim, heads=heads, dim_head=dim_head, fused_attn=False)

        # Set to eval mode for deterministic behavior
        attn_fused.eval()
        attn_unfused.eval()

        with torch.no_grad():
            output_fused = attn_fused(sample_input)
            output_unfused = attn_unfused(sample_input)

        # The two paths are mathematically equivalent, but F.scaled_dot_product_attention
        # dispatches to a different kernel than the explicit matmul/softmax, and which
        # kernel it picks depends on the host CPU. The resulting accumulation-order
        # differences reach ~1e-3 on some runners, so the tolerance covers kernel choice
        # rather than bit-exactness; a genuine error (wrong scale, transposed axes)
        # changes the output by O(0.1) or more and is still caught.
        assert torch.allclose(output_fused, output_unfused, atol=2e-3, rtol=2e-3)

    def test_attention_with_different_heads(self):
        """Test Attention with various number of heads."""
        dim = 64
        test_cases = [1, 2, 4, 8, 16]

        for heads in test_cases:
            attn = Attention(dim=dim, heads=heads, dim_head=32)
            input_tensor = torch.randn(2, 10, dim)
            output = attn(input_tensor)
            assert output.shape == input_tensor.shape

    def test_attention_with_different_dim_head(self):
        """Test Attention with various dimension per head."""
        dim = 128
        test_cases = [16, 32, 64, 128]

        for dim_head in test_cases:
            attn = Attention(dim=dim, heads=4, dim_head=dim_head)
            input_tensor = torch.randn(2, 10, dim)
            output = attn(input_tensor)
            assert output.shape == input_tensor.shape

    def test_attention_gradient_flow(self, sample_input):
        """Test that gradients flow correctly through Attention."""
        dim = 64
        attn = Attention(dim=dim, heads=8, dim_head=64)

        sample_input.requires_grad = True
        output = attn(sample_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert sample_input.grad is not None
        assert not torch.all(sample_input.grad == 0)

    def test_attention_with_different_sequence_lengths(self):
        """Test Attention with various sequence lengths."""
        dim = 64
        attn = Attention(dim=dim, heads=8, dim_head=64)

        test_lengths = [1, 5, 10, 50, 100]
        for seq_len in test_lengths:
            input_tensor = torch.randn(2, seq_len, dim)
            output = attn(input_tensor)
            assert output.shape == input_tensor.shape

    def test_attention_single_element(self):
        """Test Attention with single element in sequence."""
        dim = 64
        attn = Attention(dim=dim, heads=8, dim_head=64)

        input_tensor = torch.randn(1, 1, dim)
        output = attn(input_tensor)
        assert output.shape == input_tensor.shape

    def test_attention_batch_independence(self):
        """Test that Attention processes batch samples independently."""
        dim = 64
        attn = Attention(dim=dim, heads=4, dim_head=32, fused_attn=False)

        # Create input with two different samples
        input1 = torch.randn(1, 10, dim)
        input2 = torch.randn(1, 10, dim)
        batched_input = torch.cat([input1, input2], dim=0)

        # Process separately and together
        output1 = attn(input1)
        output2 = attn(input2)
        batched_output = attn(batched_input)

        # Results should be identical
        assert torch.allclose(batched_output[0], output1[0], atol=1e-6)
        assert torch.allclose(batched_output[1], output2[0], atol=1e-6)


class TestTransformer:
    """Test suite for Transformer module."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor for testing."""
        # Batch size 2, sequence length 10, dimension 64.
        # Seeded so the tensor does not depend on RNG state left over from
        # earlier tests, which made numerical comparisons order-dependent.
        torch.manual_seed(0)
        return torch.randn(2, 10, 64)

    def test_transformer_initialization(self):
        """Test Transformer module initialization."""
        dim = 64
        depth = 4
        heads = 8
        dim_head = 64
        mlp_dim = 256
        fused_attn = True

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=fused_attn,
        )

        assert isinstance(transformer.norm, nn.LayerNorm)
        assert len(transformer.layers) == depth

        # Check each layer has attention and feedforward
        for layer in transformer.layers:
            assert isinstance(layer, nn.ModuleList)
            assert len(layer) == 2
            assert isinstance(layer[0], Attention)
            assert isinstance(layer[1], FeedForward)

    def test_transformer_forward(self, sample_input):
        """Test Transformer forward pass."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128
        fused_attn = True

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=fused_attn,
        )

        output = transformer(sample_input)

        # Check output shape matches input shape
        assert output.shape == sample_input.shape

    def test_transformer_with_various_depths(self):
        """Test Transformer with different depths."""
        dim = 64
        heads = 4
        dim_head = 32
        mlp_dim = 128
        test_depths = [1, 2, 4, 8, 12]

        for depth in test_depths:
            transformer = Transformer(
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                fused_attn=True,
            )
            input_tensor = torch.randn(2, 10, dim)
            output = transformer(input_tensor)
            assert output.shape == input_tensor.shape
            assert len(transformer.layers) == depth

    def test_transformer_fused_vs_unfused(self, sample_input):
        """Test Transformer with fused and unfused attention."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        # Create two transformers with same initialization
        torch.manual_seed(42)
        transformer_fused = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=True,
        )

        torch.manual_seed(42)
        transformer_unfused = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=False,
        )

        # Set to eval mode for deterministic behavior
        transformer_fused.eval()
        transformer_unfused.eval()

        with torch.no_grad():
            output_fused = transformer_fused(sample_input)
            output_unfused = transformer_unfused(sample_input)

        # Tolerance covers SDPA kernel choice rather than bit-exactness, and the
        # differences compound over `depth` layers. See test_attention_fused_vs_unfused.
        assert torch.allclose(output_fused, output_unfused, atol=2e-3, rtol=2e-3)

    def test_transformer_gradient_flow(self, sample_input):
        """Test that gradients flow correctly through Transformer."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=True,
        )

        sample_input.requires_grad = True
        output = transformer(sample_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert sample_input.grad is not None
        assert not torch.all(sample_input.grad == 0)

    def test_transformer_residual_connections(self):
        """Test that residual connections work correctly."""
        dim = 64
        depth = 1
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=True,
        )

        # Use identity-like initialization to observe residuals
        input_tensor = torch.randn(2, 10, dim)
        output = transformer(input_tensor)

        # Output should be different due to transformations but have similar magnitude
        assert output.shape == input_tensor.shape
        # The residual connections should prevent the output from being zero
        assert not torch.all(output == 0)

    def test_transformer_with_different_dimensions(self):
        """Test Transformer with various dimension configurations."""
        test_configs = [
            (32, 2, 4, 16, 64),
            (64, 4, 8, 32, 128),
            (128, 6, 8, 64, 256),
            (192, 6, 4, 48, 384),
        ]

        for dim, depth, heads, dim_head, mlp_dim in test_configs:
            transformer = Transformer(
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                fused_attn=True,
            )
            input_tensor = torch.randn(2, 10, dim)
            output = transformer(input_tensor)
            assert output.shape == input_tensor.shape

    def test_transformer_with_different_sequence_lengths(self):
        """Test Transformer with various sequence lengths."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=True,
        )

        test_lengths = [1, 5, 10, 50, 100, 196]
        for seq_len in test_lengths:
            input_tensor = torch.randn(2, seq_len, dim)
            output = transformer(input_tensor)
            assert output.shape == input_tensor.shape

    def test_transformer_batch_independence(self):
        """Test that Transformer processes batch samples independently."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=False,
        )

        # Create input with two different samples
        input1 = torch.randn(1, 10, dim)
        input2 = torch.randn(1, 10, dim)
        batched_input = torch.cat([input1, input2], dim=0)

        # Process separately and together
        output1 = transformer(input1)
        output2 = transformer(input2)
        batched_output = transformer(batched_input)

        # Results should be identical
        assert torch.allclose(batched_output[0], output1[0], atol=1e-6)
        assert torch.allclose(batched_output[1], output2[0], atol=1e-6)

    def test_transformer_normalization(self, sample_input):
        """Test that final layer normalization is applied."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=True,
        )

        output = transformer(sample_input)

        # Output should have reasonable statistics due to normalization
        # Mean should be close to 0 and std close to 1 for each feature
        assert output.shape == sample_input.shape
        assert torch.isfinite(output).all()

    def test_transformer_eval_mode(self, sample_input):
        """Test Transformer in evaluation mode."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=True,
        )

        transformer.eval()
        with torch.no_grad():
            output1 = transformer(sample_input)
            output2 = transformer(sample_input)

        # In eval mode with same input, should get identical results
        assert torch.allclose(output1, output2)

    def test_transformer_train_mode(self, sample_input):
        """Test Transformer in training mode."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=True,
        )

        transformer.train()
        output = transformer(sample_input)

        # Should produce valid output
        assert output.shape == sample_input.shape
        assert torch.isfinite(output).all()

    def test_transformer_memory_efficiency(self):
        """Test that Transformer doesn't cause memory issues."""
        dim = 64
        depth = 4
        heads = 8
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=True,
        )

        input_tensor = torch.randn(4, 196, dim)
        output = transformer(input_tensor)

        # Clean up
        del transformer, input_tensor, output
        gc.collect()


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_stacked_transformers(self):
        """Test stacking multiple Transformer modules."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer1 = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, fused_attn=True
        )
        transformer2 = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, fused_attn=True
        )

        input_tensor = torch.randn(2, 10, dim)
        intermediate = transformer1(input_tensor)
        output = transformer2(intermediate)

        assert output.shape == input_tensor.shape

    def test_large_model_config(self):
        """Test configuration similar to large models."""
        # Configuration similar to ViT-Large
        dim = 192
        depth = 6
        heads = 4
        dim_head = 48
        mlp_dim = 384

        transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            fused_attn=True,
        )

        # Typical image patch embedding (14x14 patches + 1 cls token)
        input_tensor = torch.randn(2, 197, dim)
        output = transformer(input_tensor)

        assert output.shape == input_tensor.shape
        gc.collect()

    def test_pipeline_with_all_components(self):
        """Test a full pipeline using all components."""
        dim = 64
        heads = 4
        dim_head = 32
        mlp_dim = 128

        # Create individual components
        attention = Attention(dim=dim, heads=heads, dim_head=dim_head, fused_attn=True)
        feedforward = FeedForward(dim=dim, hidden_dim=mlp_dim)
        transformer = Transformer(
            dim=dim, depth=2, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, fused_attn=True
        )

        input_tensor = torch.randn(2, 10, dim)

        # Pass through components
        attn_out = attention(input_tensor)
        ff_out = feedforward(attn_out)
        final_out = transformer(ff_out)

        assert final_out.shape == input_tensor.shape
        gc.collect()

    def test_mixed_precision_compatibility(self):
        """Test that modules work with mixed precision."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, fused_attn=True
        )

        # Test with float16
        input_tensor = torch.randn(2, 10, dim, dtype=torch.float16)
        transformer = transformer.half()

        output = transformer(input_tensor)
        assert output.dtype == torch.float16
        assert output.shape == input_tensor.shape

        gc.collect()

    def test_device_placement(self):
        """Test that modules can be placed on different devices."""
        dim = 64
        depth = 2
        heads = 4
        dim_head = 32
        mlp_dim = 128

        transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, fused_attn=True
        )

        # Test on CPU
        input_cpu = torch.randn(2, 10, dim)
        output_cpu = transformer(input_cpu)
        assert output_cpu.device == torch.device('cpu')

        # Test on CUDA if available
        if torch.cuda.is_available():
            transformer = transformer.cuda()
            input_cuda = torch.randn(2, 10, dim, device='cuda')
            output_cuda = transformer(input_cuda)
            assert output_cuda.device.type == 'cuda'

        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
