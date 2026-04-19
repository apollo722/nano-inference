import torch
from nano_inference.core.request import (
    GenerateQuery,
    GenerationInputs,
    GenerationStage,
    SamplingParams,
)
from nano_inference.engine.context_builder import GenerateContextBuilder
from nano_inference.kv_cache.block import KVCacheBlock

...


def test_context_builder_builds_padded_batch():
    builder = GenerateContextBuilder(device=torch.device("cpu"))

    # Query 1: len 3
    q1 = GenerateQuery(
        request_id="q1",
        generation_inputs=GenerationInputs(prompt_token_ids=[1, 2, 3]),
        sampling_params=SamplingParams(),
        eos_token_id=0,
        arrival_time=0.0,
    )
    q1.kv_cache_block = KVCacheBlock(block_ids=[101], block_size=4)

    # Query 2: len 5 (prefill stage)
    q2 = GenerateQuery(
        request_id="q2",
        generation_inputs=GenerationInputs(prompt_token_ids=[4, 5, 6, 7, 8]),
        sampling_params=SamplingParams(),
        eos_token_id=0,
        arrival_time=0.0,
    )
    q2.kv_cache_block = KVCacheBlock(block_ids=[102, 103], block_size=4)

    context = builder.build([q1, q2])

    assert context.input_ids.shape == (2, 5)
    assert context.attention_mask.shape == (2, 5)
    assert context.position_ids.shape == (2, 5)
    assert context.request_ids == ["q1", "q2"]

    # Check Context Lens
    assert torch.equal(context.context_lens, torch.tensor([3, 5]))

    # Check Block Tables (padded to max_blocks=2)
    expected_bt = torch.tensor([[101, 0], [102, 103]], dtype=torch.int32)
    assert torch.equal(context.kv_block_tables, expected_bt)

    # Check Slot Mapping
    # q1: tokens at 0,1,2 map to block 101, offsets 0,1,2 -> indices 101*4 + 0,1,2 = 404, 405, 406
    # q2: tokens at 0,1,2,3 map to block 102 (offsets 0,1,2,3), token 4 maps to block 103 (offset 0)
    # indices: 408, 409, 410, 411, 412
    assert context.slot_mapping[0, 0] == 404
    assert context.slot_mapping[0, 2] == 406
    assert context.slot_mapping[1, 0] == 408
    assert context.slot_mapping[1, 4] == 412


...


def test_context_builder_decode_step():
    builder = GenerateContextBuilder(device=torch.device("cpu"))

    q = GenerateQuery(
        request_id="q",
        generation_inputs=GenerationInputs(prompt_token_ids=[1, 2]),
        sampling_params=SamplingParams(),
        eos_token_id=0,
        arrival_time=0.0,
    )
    q.output_token_ids = [3, 4]  # Total len 4
    q.stage = GenerationStage.DECODE
    q.kv_cache_block = KVCacheBlock(block_ids=[101], block_size=16)

    context = builder.build([q])

    # For DECODE, input_ids should only contain the LAST token
    assert context.input_ids.shape == (1, 1)
    assert context.input_ids[0, 0] == 4
    assert context.position_ids[0, 0] == 3
    assert context.context_lens[0] == 4

    # Slot mapping for token at pos 3: block 101, offset 3 -> 101*16 + 3 = 1619
    assert context.slot_mapping[0, 0] == 1619
