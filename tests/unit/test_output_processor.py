from unittest.mock import MagicMock

from nano_inference.core.request import (
    FinishedReason,
    GenerateQuery,
    GenerationInputs,
    GenerationStage,
    SamplingParams,
)
from nano_inference.driver.output_processor import OutputProcessor


def test_output_processor_appends_token_and_transitions_stage():
    mock_input_processor = MagicMock()
    mock_input_processor.decode.side_effect = (
        lambda ids, *args, **kwargs: f"text_{ids[0]}"
    )
    processor = OutputProcessor(mock_input_processor)

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
    # 1. Record step (metrics only)
    from nano_inference.scheduler.scheduler import OrcaScheduler

    scheduler = OrcaScheduler()
    scheduler.add_tasks([q])
    scheduler.record_step([q], 1)

    # Stage should STILL be PREFILL (record_step doesn't transition anymore)
    assert q.stage == GenerationStage.PREFILL

    # 2. Process outputs (transitions stage)
    processor.process_step_outputs([q], [10])

    assert q.stage == GenerationStage.DECODE

    assert q.output_token_ids == [10]
    # computed_length should be prompt_len (3) + output_len (1) = 4
    assert q.computed_length == 4
    assert q.delta_text == "text_10"
    assert q.previous_tokens_len == 1


def test_output_processor_handles_eos():
    mock_input_processor = MagicMock()
    mock_input_processor.decode.side_effect = (
        lambda ids, *args, **kwargs: f"text_{ids[0]}"
    )
    processor = OutputProcessor(mock_input_processor)

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
    mock_input_processor = MagicMock()
    mock_input_processor.decode.side_effect = (
        lambda ids, *args, **kwargs: f"text_{ids[0]}"
    )
    processor = OutputProcessor(mock_input_processor)

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
