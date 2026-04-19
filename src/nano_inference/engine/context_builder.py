from typing import List

import torch
from nano_inference.core.context import AttentionMetadata, GenerateContext
from nano_inference.core.request import GenerateQuery, GenerationStage


class GenerateContextBuilder:
    """Builds a GenerateContext from a list of active queries.

    Phase 3: Now supports Paged Attention metadata and partial input re-feeding.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def build(self, queries: List[GenerateQuery]) -> GenerateContext:
        if not queries:
            raise ValueError("Cannot build context for empty queries list.")

        # 1. Determine if this is a PREFILL or DECODE step
        # Note: Phase 2/3 currently handles batches where all are in the same stage
        # or mixed but we treat them uniformly in the context for now.
        # If any query is PREFILL, we treat the batch as prefill-heavy.
        is_prefill = any(q.stage == GenerationStage.PREFILL for q in queries)

        batch_size = len(queries)

        # 2. Extract new tokens for the current step
        all_step_token_ids = []
        for q in queries:
            if q.stage == GenerationStage.PREFILL:
                # Full prompt for prefill
                all_step_token_ids.append(q.generation_inputs.prompt_token_ids)
            else:
                # Only the last generated token for decode
                all_step_token_ids.append([q.output_token_ids[-1]])

        max_step_len = max(len(ids) for ids in all_step_token_ids)

        # 3. Build Model Inputs (Padded)
        input_ids = torch.zeros(
            (batch_size, max_step_len), dtype=torch.long, device=self.device
        )
        position_ids = torch.zeros(
            (batch_size, max_step_len), dtype=torch.long, device=self.device
        )

        # context_lens tracks the TOTAL length including what is already in cache
        context_lens = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i, (q, step_ids) in enumerate(zip(queries, all_step_token_ids)):
            step_len = len(step_ids)
            input_ids[i, :step_len] = torch.tensor(
                step_ids, dtype=torch.long, device=self.device
            )

            # Start position in the sequence
            start_pos = (
                len(q.generation_inputs.prompt_token_ids)
                + len(q.output_token_ids)
                - step_len
            )

            position_ids[i, :step_len] = torch.arange(
                start_pos, start_pos + step_len, device=self.device
            )
            context_lens[i] = start_pos + step_len

        # 4. Build Attention Mask
        # Paged Attention usually handles the mask internally using context_lens,
        # but for baseline kernels we might need a (B, S_max) mask.
        max_total_len = context_lens.max().item()
        attention_mask = torch.zeros(
            (batch_size, max_total_len), dtype=torch.bool, device=self.device
        )
        for i, length in enumerate(context_lens):
            attention_mask[i, :length] = True

        # 5. Build KV Metadata (Block Tables & Slot Mapping)
        # Find max blocks for padding
        max_blocks = max(len(q.kv_cache_block.block_ids) for q in queries)
        block_tables = torch.zeros(
            (batch_size, max_blocks), dtype=torch.int32, device=self.device
        )
        slot_mapping = torch.full(
            (batch_size, max_step_len), -1, dtype=torch.long, device=self.device
        )

        block_size = queries[0].kv_cache_block.block_size

        for i, q in enumerate(queries):
            # Populate block table
            b_ids = q.kv_cache_block.block_ids
            block_tables[i, : len(b_ids)] = torch.tensor(
                b_ids, dtype=torch.int32, device=self.device
            )

            # Populate slot mapping for tokens in input_ids
            # Each token gets a physical index: block_id * block_size + offset_in_block
            start_pos = position_ids[i, 0].item()
            for step_idx in range(len(all_step_token_ids[i])):
                curr_pos = start_pos + step_idx
                block_idx = curr_pos // block_size
                block_offset = curr_pos % block_size

                if block_idx >= len(b_ids):
                    from nano_inference.utils.logger import logger

                    logger.error(
                        f"IndexError for request {q.request_id}: block_idx {block_idx} "
                        f"out of range for {len(b_ids)} blocks. "
                        f"curr_pos={curr_pos}, block_size={block_size}"
                    )
                    # Fallback to avoid crash in this loop, but it will likely fail later
                    physical_block_id = b_ids[-1]
                else:
                    physical_block_id = b_ids[block_idx]

                slot_mapping[i, step_idx] = (
                    physical_block_id * block_size + block_offset
                )

        metadata = AttentionMetadata(
            context_lens=context_lens,
            kv_block_tables=block_tables,
            slot_mapping=slot_mapping,
            is_prefill=is_prefill,
        )

        return GenerateContext(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            metadata=metadata,
            request_ids=[q.request_id for q in queries],
        )
