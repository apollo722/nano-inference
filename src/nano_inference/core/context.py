from dataclasses import dataclass
from typing import List

import torch


@dataclass
class GenerateContext:
    """Batch-level context for a single inference step.

    Since we don't have KV cache yet (Phase 3), every step for every query
    must include the full sequence history in the input.
    """

    input_ids: torch.Tensor  # (B, S_max)
    attention_mask: torch.Tensor  # (B, S_max)
    position_ids: torch.Tensor  # (B, S_max)
    request_ids: List[str]
    query_lengths: List[int]
