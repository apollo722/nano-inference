import torch
from nano_inference.layers.attention import NaiveCausalSelfAttention


def test_attention_output_shape():
    hidden_size = 32
    num_heads = 4
    attn = NaiveCausalSelfAttention(hidden_size, num_heads)
    x = torch.randn(1, 8, hidden_size)
    position_ids = torch.arange(8).unsqueeze(0)

    out = attn(x, position_ids)
    assert out.shape == (1, 8, hidden_size)


def test_causal_self_attention_future_tokens_do_not_affect_past_outputs():
    torch.manual_seed(0)
    hidden_size = 16
    num_heads = 4
    attn = NaiveCausalSelfAttention(hidden_size, num_heads)

    x1 = torch.randn(1, 4, hidden_size, dtype=torch.float32)
    x2 = x1.clone()
    # Change future tokens
    x2[:, 2:, :] = torch.randn(1, 2, hidden_size, dtype=torch.float32)

    position_ids = torch.arange(4).unsqueeze(0)

    y1 = attn(x1, position_ids)
    y2 = attn(x2, position_ids)

    # Past outputs should be identical
    torch.testing.assert_close(y1[:, :2, :], y2[:, :2, :], atol=1e-5, rtol=1e-5)


def test_causal_self_attention_preserves_shape():
    hidden_size = 16
    num_heads = 4
    attn = NaiveCausalSelfAttention(hidden_size, num_heads)
    x = torch.randn(2, 5, hidden_size, dtype=torch.float32)
    position_ids = torch.arange(5, dtype=torch.long).unsqueeze(0).repeat(2, 1)

    y = attn(x, position_ids)

    assert y.shape == x.shape


def test_attention_gqa_output_shape():
    hidden_size = 32
    num_heads = 8
    num_kv_heads = 2
    attn = NaiveCausalSelfAttention(hidden_size, num_heads, num_kv_heads=num_kv_heads)

    x = torch.randn(1, 4, hidden_size)
    position_ids = torch.arange(4).unsqueeze(0)

    out = attn(x, position_ids)
    assert out.shape == (1, 4, hidden_size)

    # Check k_proj shape
    assert attn.k_proj.out_features == num_kv_heads * (hidden_size // num_heads)


def test_attention_qk_norm_enabled():
    hidden_size = 16
    num_heads = 2
    attn = NaiveCausalSelfAttention(hidden_size, num_heads, use_qk_norm=True)

    assert attn.q_norm is not None
    assert attn.k_norm is not None

    x = torch.randn(1, 2, hidden_size)
    position_ids = torch.arange(2).unsqueeze(0)
    out = attn(x, position_ids)
    assert out.shape == (1, 2, hidden_size)
