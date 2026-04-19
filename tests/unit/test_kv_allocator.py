import pytest
import torch
from nano_inference.kv_cache.allocator import PagedKVCacheAllocator
from nano_inference.kv_cache.block import KVCacheBlock


def test_allocator_initialization():
    num_blocks = 10
    block_size = 16
    num_heads = 4
    head_dim = 32
    num_layers = 2
    device = "cpu"

    allocator = PagedKVCacheAllocator(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        device=device,
    )

    assert allocator.num_free_blocks == num_blocks
    assert allocator.k_cache.shape == (
        num_layers,
        num_blocks,
        num_heads,
        block_size,
        head_dim,
    )
    assert allocator.v_cache.shape == (
        num_layers,
        num_blocks,
        num_heads,
        block_size,
        head_dim,
    )
    assert allocator.utilization == 0.0


def test_allocate_and_free():
    allocator = PagedKVCacheAllocator(
        num_blocks=10, block_size=16, num_heads=4, head_dim=32, device="cpu"
    )

    # Request 20 tokens -> needs 2 blocks
    block = allocator.allocate(num_tokens=20)
    assert len(block.block_ids) == 2
    assert allocator.num_free_blocks == 8
    assert allocator.utilization == pytest.approx(0.2)

    # Free blocks
    allocator.free(block)
    assert allocator.num_free_blocks == 10
    assert len(block.block_ids) == 0
    assert allocator.utilization == 0.0


def test_allocate_token_extension():
    allocator = PagedKVCacheAllocator(
        num_blocks=10, block_size=4, num_heads=2, head_dim=16, device="cpu"
    )

    # Allocate 1 block (4 tokens)
    block = allocator.allocate(num_tokens=4)
    block.append_tokens(4)
    assert block.is_full()
    assert len(block.block_ids) == 1

    # Extend
    allocator.allocate_token(block)
    assert len(block.block_ids) == 2
    assert not block.is_full()
    assert allocator.num_free_blocks == 8


def test_out_of_memory():
    allocator = PagedKVCacheAllocator(
        num_blocks=2, block_size=4, num_heads=2, head_dim=16, device="cpu"
    )

    allocator.allocate(num_tokens=8)  # Uses both blocks

    with pytest.raises(RuntimeError, match="Out of KV cache memory"):
        allocator.allocate(num_tokens=1)


def test_can_allocate():
    allocator = PagedKVCacheAllocator(
        num_blocks=5, block_size=4, num_heads=2, head_dim=16, device="cpu"
    )

    assert allocator.can_allocate(num_blocks_needed=5)
    assert not allocator.can_allocate(num_blocks_needed=6)

    allocator.allocate(num_tokens=4)  # Uses 1 block
    assert allocator.can_allocate(num_blocks_needed=4)
    assert not allocator.can_allocate(num_blocks_needed=5)
