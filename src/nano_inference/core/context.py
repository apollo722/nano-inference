from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class GenerateContext:
    """Batch-level context for a single inference step.

    Phase 3: Now supports Paged Attention metadata.
    """

    # Model Inputs
    input_ids: torch.Tensor  # (B, S_step) - S_step is 1 for decode, full for prefill
    attention_mask: torch.Tensor  # (B, S_total) - Mask for full history
    position_ids: torch.Tensor  # (B, S_step)

    # KV Cache Metadata
    context_lens: torch.Tensor  # (B,) - Total sequence length per request
    kv_block_tables: torch.Tensor  # (B, max_blocks_per_seq) - Physical block IDs
    slot_mapping: torch.Tensor  # (B, S_step) - Physical slot per token in input_ids

    # Internal Tracking
    request_ids: List[str]
    is_prefill: bool
