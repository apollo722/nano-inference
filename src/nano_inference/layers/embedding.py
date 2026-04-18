from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from nano_inference.utils.pickle_ops import dump_output_pickle


class TokenEmbeddingBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass


class NaiveTokenEmbedding(TokenEmbeddingBase):
    """
    Token Embedding layer.

    Architecture:
    1. Lookup Table: Weight matrix of size (vocab_size, hidden_size).
    2. Operation: Index-based retrieval.

    Mathematical Flow:
    - Input: input_ids of shape (batch, seq)
    - Weight: W of shape (vocab_size, hidden_size)
    - Output: output = W[input_ids]
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__(vocab_size, hidden_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

        # Standard transformer weight initialization (mean=0.0, std=0.02)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    @dump_output_pickle(name="torch-embedding")
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Input Shape: (batch, seq)
        Output Shape: (batch, seq, hidden_size)
        """
        # Efficiently lookup embedding vectors for each token ID
        return F.embedding(input_ids, self.weight)
