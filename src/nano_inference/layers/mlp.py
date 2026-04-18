from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLPBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class NaiveSwiGLUMLP(SwiGLUMLPBase):
    """
    SwiGLU MLP (Feed-Forward Network) layer.

    Architecture:
    1. Gate Projection: Input -> Intermediate
    2. Up Projection: Input -> Intermediate
    3. Activation: SiLU (Swish) applied to Gate
    4. Gating: Element-wise product of SiLU(Gate) and Up
    5. Down Projection: Intermediate -> Input

    Mathematical Flow:
    - gate = x @ W_gate
    - up = x @ W_up
    - intermediate = SiLU(gate) * up
    - output = intermediate @ W_down
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__(hidden_size, intermediate_size)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Expand: Project from model dimension to larger intermediate dimension
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # Contract: Project back to model dimension
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input Shape: (batch, seq, hidden_size)
        """
        # 1. Projections to intermediate space (batch, seq, intermediate_size)
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # 2. SwiGLU Gating: SiLU(gate) * up
        # The gating mechanism allows the model to learn non-linear filters
        hidden = F.silu(gate) * up

        # 3. Down-projection to model space (batch, seq, hidden_size)
        return self.down_proj(hidden)
