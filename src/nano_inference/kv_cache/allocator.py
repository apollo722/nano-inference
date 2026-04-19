import math
from typing import List, Optional

import torch
from nano_inference.kv_cache.block import KVCacheBlock


class PagedKVCacheAllocator:
    """Manages physical KV cache memory blocks on a device.

    This allocator maintains a pool of free blocks and provides them to
    requests as they grow. It owns the actual physical storage tensors.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)

        # 1. Initialize the free-list
        self.free_blocks: List[int] = list(range(num_blocks))

        # 2. Allocate the physical storage
        # Shape: [num_blocks, num_heads, block_size, head_dim]
        # We create separate tensors for K and V
        cache_shape = (num_blocks, num_heads, block_size, head_dim)
        self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=self.device)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=self.device)

    def can_allocate(self, num_blocks_needed: int) -> bool:
        """Check if the requested number of blocks are available."""
        return len(self.free_blocks) >= num_blocks_needed

    def allocate(self, num_tokens: int) -> KVCacheBlock:
        """Allocate blocks sufficient to hold num_tokens."""
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        if not self.can_allocate(num_blocks_needed):
            raise RuntimeError(
                f"Out of KV cache memory. Requested {num_blocks_needed} blocks, "
                f"but only {len(self.free_blocks)} available."
            )

        # Pop from the end for O(1) efficiency
        allocated_ids = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        return KVCacheBlock(block_ids=allocated_ids, block_size=self.block_size)

    def allocate_token(self, block: KVCacheBlock) -> None:
        """Allocate one additional block if the current logical block is full."""
        if block.is_full():
            if not self.free_blocks:
                raise RuntimeError("Out of KV cache memory during decode extension.")
            block.append_block(self.free_blocks.pop())

    def free(self, block: KVCacheBlock) -> None:
        """Return all physical blocks back to the free pool."""
        if block and block.block_ids:
            self.free_blocks.extend(block.block_ids)
            block.block_ids = []
            block.num_tokens = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def utilization(self) -> float:
        return 1.0 - (self.num_free_blocks / self.num_blocks)
