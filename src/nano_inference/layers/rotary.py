from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Split rotate-half: [x0, x1, x2, x3] -> [-x2, -x3, x0, x1]
    This is the standard implementation for Llama/Qwen models.
    Input/Output Shape: (batch, seq, num_heads, head_dim)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies RoPE to query and key tensors.
    Formula: q_rot = q * cos + rotate_half(q) * sin
    q/k Shapes: (batch, seq, num_heads, head_dim)
    cos/sin Shapes: (batch, seq, 1, head_dim)
    """
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RotaryEmbeddingBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class NaiveRotaryEmbedding(RotaryEmbeddingBase):
    """
    Rotary Position Embedding (RoPE).

    1. 2D Rotation Derivation:
       A pair of coordinates [x1, x2] is rotated by angle θ:
       [x1']   [cosθ  -sinθ] [x1]   [x1*cosθ - x2*sinθ]
       [x2'] = [sinθ   cosθ] [x2] = [x1*sinθ + x2*cosθ]

    2. Complex Representation:
       Representing [x1, x2] as a complex number z = x1 + ix2:
       z * e^(iθ) = (x1 + ix2)(cosθ + isinθ)
                  = (x1*cosθ - x2*sinθ) + i(x1*sinθ + x2*cosθ)

    3. Optimization:
       Instead of full matrix multiplication, RoPE leverages the rotate_half trick
       to apply rotations across head dimensions efficiently.

    θ_i = base^(-2i/d), where i = 0, 1, 2, ..., d/2 - 1
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__(head_dim, base)
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for rotary embedding, got {head_dim}"
            )

        self.head_dim = head_dim
        self.base = base

        # Precompute the inverse frequencies for the rotary embedding
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input Shapes:
            q/k: (batch, seq, num_heads, head_dim)
            position_ids: (batch, seq)
        """
        # Calculate angular frequencies for each position
        # position_ids: [B, S], inv_freq: [D/2] -> freqs: [B, S, D/2]
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.unsqueeze(
            0
        ).unsqueeze(0)

        # emb Shape: (batch, seq, head_dim)
        # Standard RoPE uses [freqs, freqs] to match [x_first_half, x_second_half]
        emb = torch.cat((freqs, freqs), dim=-1)

        # cos/sin Shape: (batch, seq, 1, head_dim)
        cos = emb.cos().to(dtype=q.dtype, device=q.device).unsqueeze(2)
        sin = emb.sin().to(dtype=q.dtype, device=q.device).unsqueeze(2)

        return apply_rotary_pos_emb(q, k, cos, sin)
