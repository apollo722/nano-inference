from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class AttentionMetadata:
    """Metadata required for Paged Attention computation."""

    # KV Cache Metadata
    context_lens: torch.Tensor  # (B,) - Total sequence length per request
    kv_block_tables: torch.Tensor  # (B, max_blocks_per_seq) - Physical block IDs
    slot_mapping: torch.Tensor  # (B, S_step) - Physical slot per token in input_ids

    # Physical Cache Tensors (optional if owned by model, but usually passed here)
    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None

    # Forward Mode
    is_prefill: bool = False


@dataclass
class GenerateContext:
    """Batch-level context for a single inference step.

    Phase 3: Now uses AttentionMetadata for Paged Attention.
    """

    # Model Inputs
    input_ids: torch.Tensor  # (B, S_step)
    attention_mask: torch.Tensor  # (B, S_total)
    position_ids: torch.Tensor  # (B, S_step)

    # All metadata for this step
    metadata: AttentionMetadata

    # Internal Tracking
    request_ids: List[str]
    # Phase 3+: Full token history per request (prompt + generated) for sampler
    token_histories: List[List[int]]
