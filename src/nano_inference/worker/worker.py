from abc import ABC, abstractmethod
from typing import List

from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerateOutput, GenerateQuery, Request
from nano_inference.inferencer.factory import InferencerFactory


class WorkerBase(ABC):
    """Abstract base for a worker that executes inference on a device.

    Interface uses List[GenerateQuery] -> List[GenerateOutput] from the start,
    so the ABC stays stable when batching is added in Phase 2.
    """

    @abstractmethod
    def generate(self, queries: List[GenerateQuery]) -> List[GenerateOutput]:
        """Run generation for a batch of queries."""
        ...


class Worker(WorkerBase):
    """Single-device worker that holds an Inferencer.

    Phase 1: iterates queries one-at-a-time (Inferencer takes a single Request).
    Phase 2+: Inferencer will accept batched input directly.
    """

    def __init__(self, inferencer_type: str, model_config: ModelConfig):
        self.inferencer = InferencerFactory.create(inferencer_type, model_config)

    def generate(self, queries: List[GenerateQuery]) -> List[GenerateOutput]:
        outputs: List[GenerateOutput] = []
        for query in queries:
            # Bridge: current Inferencer expects Request, so we reconstruct one.
            # This bridge disappears in Phase 2 when Inferencer accepts batched queries.
            request = Request(
                request_id=query.request_id,
                generation_inputs=query.generation_inputs,
                sampling_params=query.sampling_params,
                eos_token_id=query.eos_token_id,
                arrival_time=query.arrival_time,
            )
            output = self.inferencer.generate(request)
            outputs.append(output)
        return outputs
