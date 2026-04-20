import time
from unittest.mock import MagicMock

import pytest
from nano_inference.core.config import SchedulerConfig
from nano_inference.core.request import (
    GenerateQuery,
    GenerationInputs,
    GenerationStage,
    Request,
)
from nano_inference.core.sampling import SamplingParams
from nano_inference.driver.output_processor import OutputProcessor
from nano_inference.engine.context_builder import GenerateContextBuilder
from nano_inference.kv_cache.allocator import PagedKVCacheAllocator
from nano_inference.scheduler.scheduler import OrcaScheduler


def test_preemption_preserves_history():
    # Setup
    config = SchedulerConfig(max_batch_size=4, max_prefill_batch_size=1)
    # Very small allocator to trigger OOM
    allocator = PagedKVCacheAllocator(
        num_blocks=2,  # Only 2 blocks
        block_size=4,
        num_heads=1,
        head_dim=1,
        device="cpu",
    )
    scheduler = OrcaScheduler(config, allocator)

    # 1. Add a request
    sampling = SamplingParams(max_new_tokens=10)
    inputs = GenerationInputs(prompt_token_ids=[1, 2, 3])
    req = Request(
        "req1", inputs, sampling, eos_token_id=50256, arrival_time=time.time()
    )
    query = GenerateQuery.from_request(req)

    scheduler.add_tasks([query])

    # 2. Schedule PREFILL
    batch = scheduler.schedule()
    assert len(batch) == 1
    assert batch[0].stage == GenerationStage.PREFILL
    assert batch[0].kv_cache_block is not None
    assert (
        len(batch[0].kv_cache_block.block_ids) == 1
    )  # (3 prompt + 1 new) / 4 = 1 block

    # Simulate processing prefill
    output_processor = OutputProcessor(MagicMock())
    output_processor._is_eos_token = MagicMock(return_value=False)
    output_processor.input_processor.decode.return_value = "token1"

    output_processor.process_step_outputs(batch, [10])
    assert query.stage == GenerationStage.DECODE
    assert query.output_token_ids == [10]

    # 3. Simulate preemption
    scheduler.preempt_request("req1")
    assert query.stage == GenerationStage.RECOMPUTE
    assert query.output_token_ids == [10]  # History preserved!
    assert query.kv_cache_block is None  # Physical cache cleared

    # 4. Schedule RECOMPUTE
    # Now it needs Prompt(3) + Output(1) + 1 new = 5 tokens -> 2 blocks
    batch2 = scheduler.schedule()
    assert len(batch2) == 1
    assert batch2[0].stage == GenerationStage.RECOMPUTE
    assert len(batch2[0].kv_cache_block.block_ids) == 2  # 5 tokens need 2 blocks

    # 5. Build context for recompute
    builder = GenerateContextBuilder(device="cpu")
    context = builder.build(batch2)

    # Check input_ids for recompute: should be [1, 2, 3, 10]
    expected_input = [1, 2, 3, 10]
    assert context.input_ids[0].tolist() == expected_input
    assert context.metadata.is_prefill is True  # Recompute is treated as prefill

    # 6. Process recompute output
    output_processor.input_processor.decode.return_value = "token2"
    output_processor.process_step_outputs(batch2, [11])

    assert query.stage == GenerationStage.DECODE
    assert query.output_token_ids == [10, 11]  # Both tokens present
    assert query.kv_cache_block.num_tokens == 5  # 3 prompt + 2 outputs


def test_accounting_not_reset_after_preemption():
    # Setup
    config = SchedulerConfig()
    allocator = PagedKVCacheAllocator(10, 4, 1, 1, device="cpu")
    scheduler = OrcaScheduler(config, allocator)
    output_processor = OutputProcessor(MagicMock())
    output_processor._is_eos_token = MagicMock(return_value=False)

    # Request with max_new_tokens=2
    sampling = SamplingParams(max_new_tokens=2)
    query = GenerateQuery("req1", GenerationInputs([1]), sampling, 50256, time.time())

    scheduler.add_tasks([query])

    # Step 1: Prefill -> 1 token generated
    output_processor.input_processor.decode.return_value = "token1"
    batch = scheduler.schedule()
    output_processor.process_step_outputs(batch, [10])
    assert len(query.output_token_ids) == 1
    assert query.stage == GenerationStage.DECODE
    assert query.previous_tokens_len == 1

    # Step 2: Preempt
    scheduler.preempt_request("req1")
    assert len(query.output_token_ids) == 1
    assert query.previous_tokens_len == 1  # Should stay 1 to prevent re-sending

    # Step 3: Recompute -> 1 MORE token generated
    output_processor.input_processor.decode.return_value = "token2"
    batch = scheduler.schedule()
    output_processor.process_step_outputs(batch, [11])

    # Should be FINISHED now because total tokens (10, 11) == max_new_tokens (2)
    assert len(query.output_token_ids) == 2
    assert query.stage == GenerationStage.FINISHED
    assert query.finished_reason.value == "length"
    assert query.delta_text == "token2"  # Only token 11 was sent!
