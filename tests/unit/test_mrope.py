"""Unit tests for Qwen25VLLMRotaryEmbedding (mRoPE)."""

import pytest
import torch
from nano_inference.layers.rotary import NaiveRotaryEmbedding, Qwen25VLLMRotaryEmbedding


@pytest.fixture
def mrope():
    # Qwen2.5-VL-3B: head_dim=128, mrope_section=[16,24,24]
    return Qwen25VLLMRotaryEmbedding(
        head_dim=128, base=1_000_000.0, mrope_section=(16, 24, 24)
    )


@pytest.fixture
def rope_1d():
    return NaiveRotaryEmbedding(head_dim=128, base=1_000_000.0)


def test_mrope_text_matches_1d_rope(mrope, rope_1d):
    """For text tokens, all 3 mRoPE axes are equal → result must match 1D RoPE."""
    B, S, H, D = 2, 8, 4, 128
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    pos_1d = torch.arange(S).unsqueeze(0).expand(B, -1)  # (B, S)

    # 1D RoPE result
    q_1d, k_1d = rope_1d(q.clone(), k.clone(), pos_1d)

    # mRoPE with all-equal axes (text-only)
    pos_3d = pos_1d.unsqueeze(0).expand(3, -1, -1)  # (3, B, S)
    q_m, k_m = mrope(q.clone(), k.clone(), pos_3d)

    assert torch.allclose(
        q_1d, q_m, atol=1e-5
    ), "mRoPE text tokens should match 1D RoPE for q"
    assert torch.allclose(
        k_1d, k_m, atol=1e-5
    ), "mRoPE text tokens should match 1D RoPE for k"


def test_mrope_output_shape(mrope):
    B, S, H, D = 1, 16, 8, 128
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    pos_3d = torch.zeros(3, B, S, dtype=torch.long)
    q_out, k_out = mrope(q, k, pos_3d)
    assert q_out.shape == (B, S, H, D)
    assert k_out.shape == (B, S, H, D)


def test_mrope_image_differs_from_1d(mrope, rope_1d):
    """Image tokens have different t/h/w axes → mRoPE differs from 1D RoPE."""
    B, S, H, D = 1, 4, 4, 128
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)

    # Different positions per axis
    pos_t = torch.tensor([[[0, 0, 0, 0]]])  # (1, 1, 4)
    pos_h = torch.tensor([[[0, 0, 1, 1]]])
    pos_w = torch.tensor([[[0, 1, 0, 1]]])
    pos_3d = torch.cat([pos_t, pos_h, pos_w], dim=0)  # (3, 1, 4)

    pos_1d = pos_t.squeeze(0)  # (1, 4) — temporal only
    q_1d, k_1d = rope_1d(q.clone(), k.clone(), pos_1d)
    q_m, k_m = mrope(q.clone(), k.clone(), pos_3d)

    # mRoPE and 1D RoPE should differ when axes differ
    assert not torch.allclose(
        q_1d, q_m, atol=1e-5
    ), "Image mRoPE should differ from 1D RoPE"


def test_mrope_section_assertion():
    """mrope_section must sum to head_dim // 2."""
    with pytest.raises(AssertionError):
        Qwen25VLLMRotaryEmbedding(
            head_dim=128, mrope_section=(10, 10, 10)
        )  # sums to 30 ≠ 64
