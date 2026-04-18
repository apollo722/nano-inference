from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class RMSNormBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class NaiveRMSNorm(RMSNormBase):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Derivation:
    Standard LayerNorm centers the input (subtracts mean) and scales by variance.
    RMSNorm simplifies this by only scaling by the root mean square of the inputs.
    This provides similar regularization benefits while being computationally cheaper.

    Formula:
    x_norm = (x / sqrt(mean(x^2) + eps)) * weight
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability during the square and mean operations
        x_fp32 = x.float()

        # Calculate the mean of squares along the last dimension
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)

        # Apply the RMS normalization formula
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)

        # Cast back to the original input dtype and scale by the learnable weight (gamma)
        x_norm = x_norm.to(x.dtype)
        return x_norm * self.weight
