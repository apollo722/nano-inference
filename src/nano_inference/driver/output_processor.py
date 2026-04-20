from typing import Any, List, Union

from nano_inference.core.request import FinishedReason, GenerateQuery, GenerationStage


class OutputProcessor:
    """Handles the transition of queries after each inference step."""

    def __init__(self, input_processor: Any):
        self.input_processor = input_processor

    def process_step_outputs(
        self, queries: List[GenerateQuery], new_token_ids: List[int]
    ) -> None:
        """Process the output of a single inference step.

        Updates the internal state of each query based on its stage and
        the newly generated token.
        """
        assert len(queries) == len(new_token_ids)

        for query in queries:
            if not hasattr(query, "output_text_list"):
                query.output_text_list = []

        for query, token_id in zip(queries, new_token_ids):
            # 1. Update KV Cache tracking
            if query.kv_cache_block:
                if query.stage == GenerationStage.PREFILL:
                    # In prefill, we added Prompt + 1 new token
                    query.kv_cache_block.append_tokens(
                        len(query.generation_inputs.prompt_token_ids) + 1
                    )
                elif query.stage == GenerationStage.RECOMPUTE:
                    # In recompute, we added Prompt + existing outputs + 1 new token
                    query.kv_cache_block.append_tokens(
                        len(query.generation_inputs.prompt_token_ids)
                        + len(query.output_token_ids)
                        + 1
                    )
                else:
                    # In decode, we just added one token
                    query.kv_cache_block.append_tokens(1)

            # 2. Update query state with the new token
            query.output_token_ids.append(token_id)
            query.computed_length = len(query.generation_inputs.prompt_token_ids) + len(
                query.output_token_ids
            )

            # 2. Incremental detokenization
            # Decode only the NEW tokens
            new_tokens = query.output_token_ids[query.previous_tokens_len :]
            delta = self.input_processor.decode(new_tokens, skip_special_tokens=False)
            query.delta_text = delta
            query.output_text_list.append(delta)
            query.full_text = "".join(query.output_text_list)
            query.previous_tokens_len = len(query.output_token_ids)

            # 3. Transition to DECODE if we just finished a PREFILL or RECOMPUTE step
            if query.stage in (GenerationStage.PREFILL, GenerationStage.RECOMPUTE):
                query.stage = GenerationStage.DECODE

            # 4. Check for stop conditions
            if self._is_eos_token(token_id, query.eos_token_id):
                query.stage = GenerationStage.FINISHED
                query.finished_reason = FinishedReason.STOP
                continue

            if (
                query.sampling_params.max_new_tokens
                and len(query.output_token_ids) >= query.sampling_params.max_new_tokens
            ):
                query.stage = GenerationStage.FINISHED
                query.finished_reason = FinishedReason.LENGTH
                continue

    @staticmethod
    def _is_eos_token(token_id: int, eos_token_id: Union[int, List[int]]) -> bool:
        if isinstance(eos_token_id, int):
            return token_id == eos_token_id
        return token_id in eos_token_id
