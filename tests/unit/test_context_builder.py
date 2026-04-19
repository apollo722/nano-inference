import torch
from nano_inference.core.request import GenerateQuery, GenerationInputs, SamplingParams
from nano_inference.engine.context_builder import GenerateContextBuilder


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
    # Query 2: len 5 (prefill stage)
    q2 = GenerateQuery(
        request_id="q2",
        generation_inputs=GenerationInputs(prompt_token_ids=[4, 5, 6, 7, 8]),
        sampling_params=SamplingParams(),
        eos_token_id=0,
        arrival_time=0.0,
    )

    context = builder.build([q1, q2])

    assert context.input_ids.shape == (2, 5)
    assert context.attention_mask.shape == (2, 5)
    assert context.position_ids.shape == (2, 5)
    assert context.request_ids == ["q1", "q2"]
    assert context.query_lengths == [3, 5]

    # Check padding for q1
    assert torch.equal(context.input_ids[0, :3], torch.tensor([1, 2, 3]))
    assert torch.equal(context.input_ids[0, 3:], torch.tensor([0, 0]))
    assert torch.equal(context.attention_mask[0, :3], torch.tensor([True, True, True]))
    assert torch.equal(context.attention_mask[0, 3:], torch.tensor([False, False]))

    # Check q2 (no padding)
    assert torch.equal(context.input_ids[1], torch.tensor([4, 5, 6, 7, 8]))
    assert torch.equal(context.attention_mask[1], torch.tensor([True] * 5))


def test_context_builder_includes_output_tokens():
    builder = GenerateContextBuilder(device=torch.device("cpu"))

    q = GenerateQuery(
        request_id="q",
        generation_inputs=GenerationInputs(prompt_token_ids=[1, 2]),
        sampling_params=SamplingParams(),
        eos_token_id=0,
        arrival_time=0.0,
    )
    q.output_token_ids = [3, 4]  # 2 prompt + 2 output = 4 total

    context = builder.build([q])

    assert context.input_ids.shape == (1, 4)
    assert torch.equal(context.input_ids[0], torch.tensor([1, 2, 3, 4]))
    assert context.query_lengths == [4]
