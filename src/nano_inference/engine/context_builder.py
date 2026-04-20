from typing import Any, List, Optional, Tuple

import torch
from nano_inference.core.context import AttentionMetadata, GenerateContext
from nano_inference.core.request import GenerateQuery, GenerationStage


class GenerateContextBuilder:
    """Builds a GenerateContext from a list of active queries.

    Phase 3: Now supports Paged Attention metadata and partial input re-feeding.
    Phase 4: Extended with mRoPE position IDs and VLM image propagation.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def _build_mrope_for_batch(
        self,
        queries: List[GenerateQuery],
        all_step_token_ids: List[List[int]],
        position_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Build (3, B, S_step) mRoPE position tensor if any query uses VLM.

        Returns None if no query has mrope_position_ids (text-only batch).
        """
        has_mrope = any(
            q.generation_inputs.mrope_position_ids is not None for q in queries
        )
        if not has_mrope:
            return None

        batch_size = len(queries)
        max_step_len = position_ids.shape[1]
        mrope = torch.zeros(
            (3, batch_size, max_step_len), dtype=torch.long, device=self.device
        )

        for i, (q, step_ids) in enumerate(zip(queries, all_step_token_ids)):
            step_len = len(step_ids)
            q_mrope = q.generation_inputs.mrope_position_ids

            if q.stage == GenerationStage.PREFILL:
                # Prompt: use precomputed mrope (3, prompt_len)
                assert q_mrope is not None
                mrope[:, i, :step_len] = q_mrope[:, :step_len].to(self.device)

            elif q.stage == GenerationStage.RECOMPUTE:
                # Prompt portion from precomputed mrope; output portion is text → same pos all axes
                prompt_len = len(q.generation_inputs.prompt_token_ids)
                output_ids = q.output_token_ids
                if q_mrope is not None:
                    mrope[:, i, :prompt_len] = q_mrope[:, :prompt_len].to(self.device)
                    # Output tokens are text; continue positions from end of prompt mrope
                    last_pos = int(q_mrope[0, -1].item())
                    for j, _ in enumerate(output_ids):
                        p = last_pos + j + 1
                        mrope[0, i, prompt_len + j] = p
                        mrope[1, i, prompt_len + j] = p
                        mrope[2, i, prompt_len + j] = p
                else:
                    # Text-only recompute: all 3 axes = 1D position
                    for j in range(step_len):
                        p = int(position_ids[i, j].item())
                        mrope[0, i, j] = p
                        mrope[1, i, j] = p
                        mrope[2, i, j] = p

            else:  # DECODE
                # New text token: all 3 axes = same scalar
                if q_mrope is not None:
                    last_prompt_pos = int(q_mrope[0, -1].item())
                    decode_pos = last_prompt_pos + len(q.output_token_ids)
                else:
                    decode_pos = int(position_ids[i, 0].item())
                mrope[0, i, 0] = decode_pos
                mrope[1, i, 0] = decode_pos
                mrope[2, i, 0] = decode_pos

        return mrope

    def build(self, queries: List[GenerateQuery]) -> GenerateContext:
        if not queries:
            raise ValueError("Cannot build context for empty queries list.")

        # 1. Determine if this is a PREFILL/RECOMPUTE or DECODE step
        is_prefill = any(
            q.stage in (GenerationStage.PREFILL, GenerationStage.RECOMPUTE)
            for q in queries
        )

        batch_size = len(queries)

        # 2. Extract new tokens for the current step
        all_step_token_ids = []
        for q in queries:
            if q.stage == GenerationStage.PREFILL:
                # Full prompt for prefill
                all_step_token_ids.append(q.generation_inputs.prompt_token_ids)
            elif q.stage == GenerationStage.RECOMPUTE:
                # Prompt + existing outputs to rebuild cache
                all_step_token_ids.append(
                    q.generation_inputs.prompt_token_ids + q.output_token_ids
                )
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

        # 6. mRoPE position IDs (VLM only; None for text-only batches)
        mrope_position_ids = self._build_mrope_for_batch(
            queries, all_step_token_ids, position_ids
        )

        metadata = AttentionMetadata(
            context_lens=context_lens,
            kv_block_tables=block_tables,
            slot_mapping=slot_mapping,
            is_prefill=is_prefill,
            mrope_position_ids=mrope_position_ids,
        )

        # 7. Build Token Histories for Repetition Penalty
        token_histories = []
        for q in queries:
            token_histories.append(
                q.generation_inputs.prompt_token_ids + q.output_token_ids
            )

        # 8. Collect images and grid info for VLM prefill pass
        context_images: Optional[List[Any]] = None
        context_image_grid_thw: Optional[List[Tuple[int, int, int]]] = None
        if is_prefill:
            all_images: List[Any] = []
            all_grids: List[Tuple[int, int, int]] = []
            for q in queries:
                if (
                    q.stage in (GenerationStage.PREFILL, GenerationStage.RECOMPUTE)
                    and q.generation_inputs.images
                ):
                    all_images.extend(q.generation_inputs.images)
                    if q.generation_inputs.image_grid_thw:
                        all_grids.extend(q.generation_inputs.image_grid_thw)
            if all_images:
                context_images = all_images
                context_image_grid_thw = all_grids or None

        return GenerateContext(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            metadata=metadata,
            request_ids=[q.request_id for q in queries],
            token_histories=token_histories,
            images=context_images,
            image_grid_thw=context_image_grid_thw,
        )
