from nano_inference.core.request import (
    FinishedReason,
    GenerateQuery,
    GenerationInputs,
    GenerationStage,
    SamplingParams,
)
from nano_inference.driver.output_processor import OutputProcessor


def test_output_processor_appends_token_and_transitions_stage():
    processor = OutputProcessor()

    q = GenerateQuery(
        request_id="q1",
        generation_inputs=GenerationInputs(prompt_token_ids=[1, 2, 3]),
        sampling_params=SamplingParams(),
        eos_token_id=0,
        arrival_time=0.0,
    )
    assert q.stage == GenerationStage.PREFILL
    assert q.output_token_ids == []

    # After first step (prefill complete)
    processor.process_step_outputs([q], [10])

    assert q.stage == GenerationStage.DECODE
    assert q.output_token_ids == [10]
    assert q.computed_length == 1


def test_output_processor_handles_eos():
    processor = OutputProcessor()

    q = GenerateQuery(
        request_id="q1",
        generation_inputs=GenerationInputs(prompt_token_ids=[1, 2, 3]),
        sampling_params=SamplingParams(),
        eos_token_id=0,
        arrival_time=0.0,
    )
    q.stage = GenerationStage.DECODE

    # EOS token is 0
    processor.process_step_outputs([q], [0])

    assert q.stage == GenerationStage.FINISHED
    assert q.finished_reason == FinishedReason.STOP
    assert q.output_token_ids == [0]


def test_output_processor_handles_max_new_tokens():
    processor = OutputProcessor()

    q = GenerateQuery(
        request_id="q1",
        generation_inputs=GenerationInputs(prompt_token_ids=[1, 2, 3]),
        sampling_params=SamplingParams(max_new_tokens=2),
        eos_token_id=0,
        arrival_time=0.0,
    )
    q.stage = GenerationStage.DECODE

    # First token
    processor.process_step_outputs([q], [10])
    assert q.stage == GenerationStage.DECODE

    # Second token (reach max_new_tokens)
    processor.process_step_outputs([q], [11])
    assert q.stage == GenerationStage.FINISHED
    assert q.finished_reason == FinishedReason.LENGTH
    assert q.output_token_ids == [10, 11]
