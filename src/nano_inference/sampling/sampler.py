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

    def select_batch(
        self,
        logits: torch.Tensor,
        all_generated_ids: List[List[int]],
        all_sampling_params: List[SamplingParams],
    ) -> List[int]:
        """Select next tokens for a batch of logits using vectorized operations.

        Note: Currently applies same parameters per batch for absolute vectorization
        but could be extended for per-row parameters.
        """
        batch_size = logits.shape[0]
        assert batch_size == len(all_generated_ids) == len(all_sampling_params)

        # 1. Apply repetition penalty (batched if possible, or loop for now as it's complex)
        for i in range(batch_size):
            logits[i] = self._apply_repetition_penalty(
                logits[i],
                all_generated_ids[i],
                all_sampling_params[i].repetition_penalty,
            )

        # 2. Greedy sampling (temperature=0)
        # We need to handle mixed temperature in a batch
        temperatures = torch.tensor(
            [p.temperature for p in all_sampling_params],
            device=logits.device,
            dtype=logits.dtype,
        )
        greedy_mask = temperatures == 0

        next_token_ids = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        if greedy_mask.any():
            next_token_ids[greedy_mask] = torch.argmax(logits[greedy_mask], dim=-1)

        # 3. Random sampling
        if (~greedy_mask).any():
            sample_logits = logits[~greedy_mask] / temperatures[~greedy_mask].unsqueeze(
                -1
            )

            # Apply top-k/top-p (currently loops per param set, can be vectorized further if parameters match)
            for idx, i in enumerate(torch.where(~greedy_mask)[0].tolist()):
                sample_logits[idx] = self._apply_top_k_top_p(
                    sample_logits[idx],
                    top_k=all_sampling_params[i].top_k,
                    top_p=all_sampling_params[i].top_p,
                )

            probs = torch.softmax(sample_logits, dim=-1)
            next_token_ids[~greedy_mask] = torch.multinomial(
                probs, num_samples=1
            ).squeeze(-1)

        return next_token_ids.tolist()

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
