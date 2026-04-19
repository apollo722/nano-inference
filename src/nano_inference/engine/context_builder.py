from typing import List

import torch
from nano_inference.core.context import GenerateContext
from nano_inference.core.request import GenerateQuery


class GenerateContextBuilder:
    """Builds a GenerateContext from a list of active queries."""

    def __init__(self, device: torch.device):
        self.device = device

    def build(self, queries: List[GenerateQuery]) -> GenerateContext:
        if not queries:
            raise ValueError("Cannot build context for empty queries list.")

        # Gather all tokens (history + current) for every query
        # Since we don't have KV cache yet, we re-feed everything.
        all_token_sequences = []
        for q in queries:
            full_seq = q.generation_inputs.prompt_token_ids + q.output_token_ids
            all_token_sequences.append(full_seq)

        max_len = max(len(seq) for seq in all_token_sequences)
        batch_size = len(queries)

        # Build tensors with padding (right-padding for now)
        input_ids = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.bool, device=self.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )

        for i, seq in enumerate(all_token_sequences):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(
                seq, dtype=torch.long, device=self.device
            )
            attention_mask[i, :seq_len] = True
            position_ids[i, :seq_len] = torch.arange(seq_len, device=self.device)

        return GenerateContext(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            request_ids=[q.request_id for q in queries],
            query_lengths=[len(seq) for seq in all_token_sequences],
        )
