import torch
import pytest
from nano_inference.layers.norm import NaiveRMSNorm

def test_rms_norm_output_shape():
    hidden_size = 16
    norm = NaiveRMSNorm(hidden_size)
    x = torch.randn(2, 4, hidden_size)
    out = norm(x)
    assert out.shape == x.shape

def test_rms_norm_normalization():
    hidden_size = 64
    eps = 1e-6
    norm = NaiveRMSNorm(hidden_size, eps=eps)
    x = torch.randn(1, 1, hidden_size) * 10.0 # Large values
    out = norm(x)
    
    # Check if RMS is close to 1 (before scaling by weight)
    # Since weight is initialized to 1s, it should be normalized
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)

def test_rmsnorm_matches_reference_formula():
    hidden_size = 4
    norm = NaiveRMSNorm(hidden_size, eps=1e-6)
    x = torch.arange(1.0, 5.0, 1, dtype=torch.float32)
    y = norm(x)
    
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    expected = x_fp32 * torch.rsqrt(variance + 1e-6)
    expected = expected.to(x.dtype) * norm.weight

    torch.testing.assert_close(y, expected)
