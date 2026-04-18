from __future__ import annotations

from typing import List

import torch
from nano_inference.core.sampling import SamplingParams
from nano_inference.sampling.base import SamplerBase


class Sampler(SamplerBase):
    def select(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        sampling_params: SamplingParams,
    ) -> int:
        logits = self._apply_repetition_penalty(
            logits,
            generated_ids,
            sampling_params.repetition_penalty,
        )

        if sampling_params.temperature == 0:
            return int(torch.argmax(logits).item())

        logits = logits / sampling_params.temperature
        logits = self._apply_top_k_top_p(
            logits,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: List[int],
        repetition_penalty: float,
    ) -> torch.Tensor:
        if repetition_penalty == 1.0 or not generated_ids:
            return logits

        adjusted = logits.clone()
        for token_id in set(generated_ids):
            if adjusted[token_id] < 0:
                adjusted[token_id] *= repetition_penalty
            else:
                adjusted[token_id] /= repetition_penalty
        return adjusted

    @staticmethod
    def _apply_top_k_top_p(
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        filtered = logits.clone()

        if top_k > 0 and top_k < filtered.numel():
            threshold = torch.topk(filtered, top_k).values[..., -1]
            filtered[filtered < threshold] = float("-inf")

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs > top_p
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False

            filtered[sorted_indices[sorted_mask]] = float("-inf")

        return filtered