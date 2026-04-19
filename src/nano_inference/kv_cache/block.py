from typing import List


class KVCacheBlock:
    """Logical representation of a sequence's KV cache memory.

    This object holds the IDs of the physical blocks allocated to a specific
    GenerateQuery and tracks how many tokens have been written into them.
    """

    def __init__(self, block_ids: List[int], block_size: int):
        self.block_ids = block_ids
        self.block_size = block_size
        self.num_tokens = 0

    def is_full(self) -> bool:
        """Checks if the currently allocated blocks are completely filled."""
        return self.num_tokens >= len(self.block_ids) * self.block_size

    def append_block(self, block_id: int):
        """Add a new physical block to the logical sequence."""
        self.block_ids.append(block_id)

    def append_tokens(self, count: int = 1):
        """Increment the token count after writing to the cache."""
        self.num_tokens += count

    @property
    def capacity(self) -> int:
        """Total token capacity of the currently allocated blocks."""
        return len(self.block_ids) * self.block_size

    @property
    def free_slots(self) -> int:
        """Number of remaining token slots in the current blocks."""
        return self.capacity - self.num_tokens
