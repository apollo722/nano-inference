from abc import ABC, abstractmethod
from typing import List

import torch
from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerateOutput, GenerateQuery
from nano_inference.engine.context_builder import GenerateContextBuilder
from nano_inference.inferencer.factory import InferencerFactory


class WorkerBase(ABC):
    """Abstract base for a worker that executes inference on a device.

    Interface uses List[GenerateQuery] -> List[GenerateOutput] from the start,
    so the ABC stays stable when batching is added in Phase 2.
    """

    @abstractmethod
    def step(self, queries: List[GenerateQuery]) -> List[int]:
        """Run a single inference step for a batch of queries."""
        ...


class Worker(WorkerBase):
    """Single-device worker that holds an Inferencer.

    Phase 1: iterates queries one-at-a-time (Inferencer takes a single Request).
    Phase 2+: Inferencer will accept batched input directly.
    """

    def __init__(self, inferencer_type: str, model_config: ModelConfig):
        self.inferencer = InferencerFactory.create(inferencer_type, model_config)
        self.device = torch.device(model_config.device)
        self.context_builder = GenerateContextBuilder(self.device)

    def step(self, queries: List[GenerateQuery]) -> List[int]:
        context = self.context_builder.build(queries)
        sampling_params = [q.sampling_params for q in queries]
        return self.inferencer.step(context, sampling_params)
