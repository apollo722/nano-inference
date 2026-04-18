import torch
import pytest
from nano_inference.layers.rotary import NaiveRotaryEmbedding

def test_rotary_embedding_output_shape():
    head_dim = 16
    rotary = NaiveRotaryEmbedding(head_dim)
    q = torch.randn(1, 4, 2, head_dim) # [bs, seq, num_heads, head_dim]
    k = torch.randn(1, 4, 2, head_dim)
    position_ids = torch.tensor([[0, 1, 2, 3]])
    
    q_out, k_out = rotary(q, k, position_ids)
    
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

def test_rotary_embedding_rotates_vectors():
    # RoPE should preserve norm
    head_dim = 8
    rotary = NaiveRotaryEmbedding(head_dim)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    position_ids = torch.tensor([[5]])
    
    q_out, k_out = rotary(q, k, position_ids)
    
    torch.testing.assert_close(q.norm(), q_out.norm())
    torch.testing.assert_close(k.norm(), k_out.norm())

def test_rotary_embedding_position_zero_is_identity():
    head_dim = 8
    rotary = NaiveRotaryEmbedding(head_dim)
    q = torch.randn(1, 1, 2, head_dim, dtype=torch.float32)
    k = torch.randn(1, 1, 2, head_dim, dtype=torch.float32)
    position_ids = torch.zeros((1, 1), dtype=torch.float32)

    q_out, k_out = rotary(q, k, position_ids)

    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)

def test_rotary_embedding_fails_on_odd_head_dim():
    with pytest.raises(ValueError, match="head_dim must be even"):
        NaiveRotaryEmbedding(head_dim=7)
