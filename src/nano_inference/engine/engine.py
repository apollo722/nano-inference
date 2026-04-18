from abc import ABC, abstractmethod
from typing import List

from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerateOutput, GenerateQuery
from nano_inference.worker import Worker


class EngineBase(ABC):
    """Abstract base for the engine layer.

    The engine orchestrates one or more workers. In Phase 1 it's a thin
    pass-through to a single Worker. In Phase 7 (TP) it becomes the
    parallelism orchestrator that manages a WorkerGroup.
    """

    @abstractmethod
    def generate(self, queries: List[GenerateQuery]) -> List[GenerateOutput]:
        """Dispatch queries to worker(s) and collect outputs."""
        ...


class SingleWorkerEngine(EngineBase):
    """Engine backed by a single Worker on one device.

    Phase 7+ will introduce TPEngine / WorkerGroup for multi-GPU.
    """

    def __init__(self, inferencer_type: str, model_config: ModelConfig):
        self.worker = Worker(inferencer_type, model_config)

    def generate(self, queries: List[GenerateQuery]) -> List[GenerateOutput]:
        return self.worker.generate(queries)
