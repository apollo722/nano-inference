from abc import ABC, abstractmethod
from typing import List

import torch
from nano_inference.core.config import ModelConfig, RuntimeConfig
from nano_inference.core.request import GenerateOutput, GenerateQuery
from nano_inference.engine.context_builder import GenerateContextBuilder
from nano_inference.inferencer.factory import InferencerFactory
from nano_inference.kv_cache import PagedKVCacheAllocator
from nano_inference.utils.logger import logger


class WorkerBase(ABC):
    """Abstract base for a worker that executes inference on a device.

    Interface uses List[GenerateQuery] -> List[GenerateOutput] from the start,
    so the ABC stays stable when batching is added in Phase 2.
    """

    @abstractmethod
    def generate(self, queries: List[GenerateQuery]) -> List[GenerateOutput]:
        """Run full generation for a batch of queries."""
        ...

    @abstractmethod
    def step(self, queries: List[GenerateQuery]) -> List[int]:
        """Run a single inference step for a batch of queries."""
        ...


class Worker(WorkerBase):
    """Single-device worker that holds an Inferencer.

    Phase 1: iterates queries one-at-a-time (Inferencer takes a single Request).
    Phase 2+: Inferencer will accept batched input directly.
    """

    def __init__(
        self,
        inferencer_type: str,
        model_config: ModelConfig,
        allocator: PagedKVCacheAllocator = None,
    ):
        self.inferencer = InferencerFactory.create(inferencer_type, model_config)
        self.device = torch.device(model_config.device)
        self.context_builder = GenerateContextBuilder(self.device)
        self.allocator = allocator

    def init_cache(self, config: RuntimeConfig) -> None:
        """Initialize the KV cache allocator with dynamic profiling."""
        if self.allocator is not None:
            return

        model = self.inferencer.model
        num_kv_heads = (
            getattr(model.config, "num_kv_heads", None) or model.config.num_heads
        )
        head_dim = getattr(model.config, "head_dim", None) or (
            model.config.hidden_size // model.config.num_heads
        )

        num_blocks = self.profile_num_blocks(
            config, num_kv_heads, head_dim, next(model.parameters()).dtype
        )

        self.allocator = PagedKVCacheAllocator(
            num_blocks=num_blocks,
            block_size=config.kv_cache.block_size,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=next(model.parameters()).dtype,
            device=str(self.device),
        )
        logger.info(f"[Worker] Initialized PagedKVCache with {num_blocks} blocks.")

    def profile_num_blocks(
        self,
        config: RuntimeConfig,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        """Calculate the number of blocks that can fit in memory."""
        if config.kv_cache.num_blocks_gpu is not None and self.device.type == "cuda":
            return config.kv_cache.num_blocks_gpu

        if self.device.type == "cpu":
            return config.kv_cache.num_blocks_cpu

        # GPU Profiling
        torch.cuda.empty_cache()
        total_mem, free_mem = torch.cuda.mem_get_info(self.device)

        # Available memory for KV cache
        kv_mem_limit = free_mem * config.kv_cache.gpu_memory_utilization

        # Size of one block (K + V) in bytes
        element_size = torch.tensor([], dtype=dtype).element_size()
        # 2 for K and V
        block_size_bytes = (
            2 * num_kv_heads * config.kv_cache.block_size * head_dim * element_size
        )

        num_blocks = int(kv_mem_limit // block_size_bytes)
        if num_blocks <= 0:
            raise RuntimeError(
                f"Not enough GPU memory to allocate even one KV block! Needed {block_size_bytes} bytes."
            )

        return num_blocks

    def generate(self, queries: List[GenerateQuery]) -> List[GenerateOutput]:
        """Monolithic generation path for baselines or single-request drivers."""
        from nano_inference.core.request import Request

        requests = [
            Request(
                request_id=q.request_id,
                generation_inputs=q.generation_inputs,
                sampling_params=q.sampling_params,
                eos_token_id=q.eos_token_id,
                arrival_time=q.arrival_time,
            )
            for q in queries
        ]
        return self.inferencer.generate_batch(requests)

    def step(self, queries: List[GenerateQuery]) -> List[int]:
        context = self.context_builder.build(queries)
        sampling_params = [q.sampling_params for q in queries]

        # Pass physical KV tensors to the inferencer
        k_cache = self.allocator.k_cache if self.allocator else None
        v_cache = self.allocator.v_cache if self.allocator else None

        return self.inferencer.step(
            context, sampling_params, k_cache=k_cache, v_cache=v_cache
        )
