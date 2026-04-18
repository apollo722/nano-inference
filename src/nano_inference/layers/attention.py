import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from nano_inference.layers.norm import NaiveRMSNorm
from nano_inference.layers.rotary import NaiveRotaryEmbedding


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention.

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    head_dim = q.shape[-1]

    # Compute scaled similarity scores
    # (B, H, S, S) = (B, H, S, D) @ (B, H, D, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply causal mask to prevent attention to future tokens
    if is_causal:
        query_len = q.shape[-2]
        key_len = k.shape[-2]
        causal_mask = torch.triu(
            torch.ones(query_len, key_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Apply external attention mask (e.g., for padding)
    if attention_mask is not None:
        scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)

    # Normalize scores to probabilities along the last dimension
    attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    # Compute final attention output as weighted sum of values
    # (B, H, S, D) = (B, H, S, S) @ (B, H, S, D)
    return torch.matmul(attn_weights, v)


class CausalSelfAttentionBase(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_qk_norm: bool = False,
        qkv_bias: bool = False,
        out_bias: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass


class NaiveCausalSelfAttention(CausalSelfAttentionBase):
    """
    Causal Self-Attention layer supporting Grouped-Query Attention (GQA).

    Architecture:
    1. Project input to Q, K, V.
    2. (Optional) Apply QK-Normalization.
    3. Apply Rotary Positional Embeddings (RoPE) to Q and K.
    4. Repeat K, V heads if using GQA.
    5. Compute Scaled Dot-Product Attention.
    6. Project concatenated head outputs back to hidden_size.

    Mathematical flow:
    - Q = xW_q, K = xW_k, V = xW_v
    - Q, K = RoPE(Q, K, position_ids)
    - Output = Softmax(QK^T / sqrt(d_k))V * W_o
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_qk_norm: bool = False,
        qkv_bias: bool = False,
        out_bias: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            use_qk_norm,
            qkv_bias,
            out_bias,
            rope_base,
        )

        if head_dim is None and hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, got {hidden_size=} and {num_heads=}"
            )

        if num_kv_heads is None:
            num_kv_heads = num_heads

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads must be divisible by num_kv_heads, got {num_heads=} and {num_kv_heads=}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.use_qk_norm = use_qk_norm

        # Projections: Input -> Q, K, V
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias
        )
        # Output projection back to model dimension
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=out_bias,
        )

        # Optional QK-Normalization (used in Qwen3)
        self.q_norm = NaiveRMSNorm(self.head_dim) if use_qk_norm else None
        self.k_norm = NaiveRMSNorm(self.head_dim) if use_qk_norm else None
        self.rotary = NaiveRotaryEmbedding(head_dim=self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input Shapes:
            x: (batch, seq, hidden_size)
            position_ids: (batch, seq)
            attention_mask: (batch, seq, seq)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to Q, K, V and reshape to (B, S, num_heads, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 2. Apply optional QK normalization
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # 3. Apply Rotary Positional Embeddings
        q, k = self.rotary(q, k, position_ids)

        # 4. GQA: Repeat KV heads to match Q heads if necessary
        if self.num_kv_heads != self.num_heads:
            num_repeats = self.num_heads // self.num_kv_heads
            # Shape: (B, S, num_heads, head_dim)
            k = k.repeat_interleave(num_repeats, dim=2)
            v = v.repeat_interleave(num_repeats, dim=2)

        # 5. Transpose to (B, H, S, D) for optimized attention kernels
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 6. Compute scaled dot-product attention
        # Shape: (B, num_heads, S, head_dim)
        attn_out = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            attention_mask=attention_mask,
            is_causal=True,
        )

        # 7. Reshape and project back to hidden_size
        # Shape: (B, S, hidden_size)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )

        return self.o_proj(attn_out)
