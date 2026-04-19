from abc import ABC, abstractmethod
from typing import List

from nano_inference.core.config import ModelConfig, RuntimeConfig
from nano_inference.core.request import GenerateOutput, GenerateQuery
from nano_inference.kv_cache import PagedKVCacheAllocator
from nano_inference.worker import Worker


class EngineBase(ABC):
    """Abstract base for the engine layer.

    The engine orchestrates one or more workers. In Phase 1 it's a thin
    pass-through to a single Worker. In Phase 7 (TP) it becomes the
    parallelism orchestrator that manages a WorkerGroup.
    """

    @abstractmethod
    def generate(self, queries: List[GenerateQuery]) -> List[GenerateOutput]:
        """Dispatch queries to worker(s) and collect full outputs."""
        ...

    @abstractmethod
    def step(self, queries: List[GenerateQuery]) -> List[int]:
        """Run a single inference step for a batch of queries."""
        ...

    @abstractmethod
    def init_cache(self, config: RuntimeConfig) -> None:
        """Initialize the KV cache (e.g. after model load)."""
        ...


class SingleWorkerEngine(EngineBase):
    """Engine backed by a single Worker on one device.

    Phase 7+ will introduce TPEngine / WorkerGroup for multi-GPU.
    """

    def __init__(
        self,
        inferencer_type: str,
        model_config: ModelConfig,
        allocator: PagedKVCacheAllocator = None,
    ):
        self.worker = Worker(inferencer_type, model_config, allocator=allocator)

    def generate(self, queries: List[GenerateQuery]) -> List[GenerateOutput]:
        return self.worker.generate(queries)

    def step(self, queries: List[GenerateQuery]) -> List[int]:
        return self.worker.step(queries)

    def init_cache(self, config: RuntimeConfig) -> None:
        self.worker.init_cache(config)
