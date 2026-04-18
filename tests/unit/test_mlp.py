import torch
from nano_inference.layers.mlp import NaiveSwiGLUMLP


def test_mlp_output_shape():
    hidden_size = 16
    intermediate_size = 32
    mlp = NaiveSwiGLUMLP(hidden_size, intermediate_size)
    x = torch.randn(2, 4, hidden_size)
    out = mlp(x)
    assert out.shape == (2, 4, hidden_size)


def test_mlp_intermediate_projection():
    hidden_size = 16
    intermediate_size = 64
    mlp = NaiveSwiGLUMLP(hidden_size, intermediate_size)
    # Verify weights shapes
    assert mlp.gate_proj.weight.shape == (intermediate_size, hidden_size)
    assert mlp.up_proj.weight.shape == (intermediate_size, hidden_size)
    assert mlp.down_proj.weight.shape == (hidden_size, intermediate_size)


def test_swiglu_mlp_matches_manual_formula():
    import torch.nn.functional as F

    hidden_size = 4
    intermediate_size = 8
    mlp = NaiveSwiGLUMLP(hidden_size, intermediate_size)
    x = torch.randn(2, 3, hidden_size, dtype=torch.float32)

    y = mlp(x)

    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    expected = mlp.down_proj(F.silu(gate) * up)

    torch.testing.assert_close(y, expected)


def test_swiglu_mlp_preserves_batch_and_sequence_shape():
    hidden_size = 16
    intermediate_size = 32
    mlp = NaiveSwiGLUMLP(hidden_size, intermediate_size)
    x = torch.randn(2, 5, hidden_size, dtype=torch.float32)

    y = mlp(x)

    assert y.shape == x.shape
