from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from nano_inference.utils.pickle_ops import dump_output_pickle


class LogitsProcessorBase(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        embed_tokens_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass


class NaiveLogitsProcessor(LogitsProcessorBase):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tie_word_embeddings: bool = False,
    ):
        super().__init__(hidden_size, vocab_size, tie_word_embeddings)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings

        # Only create lm_head if not tying embeddings
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @dump_output_pickle("torch-logits-processor")
    def forward(
        self,
        hidden_states: torch.Tensor,
        embed_tokens_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.tie_word_embeddings:
            assert (
                embed_tokens_weight is not None
            ), "embed_tokens_weight must be provided when tie_word_embeddings=True"
            return torch.nn.functional.linear(
                hidden_states,
                weight=embed_tokens_weight,
                bias=None,
            )
        else:
            return self.lm_head(hidden_states)
