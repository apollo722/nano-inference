from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch
from nano_inference.core.sampling import SamplingParams


class SamplerBase(ABC):
    @abstractmethod
    def select(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        sampling_params: SamplingParams,
    ) -> int: ...

    @abstractmethod
    def select_batch(
        self,
        logits: torch.Tensor,
        all_generated_ids: List[List[int]],
        all_sampling_params: List[SamplingParams],
    ) -> List[int]: ...
