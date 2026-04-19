import asyncio
import time
from unittest.mock import MagicMock

import pytest
from nano_inference.core.config import ModelConfig, SchedulerConfig
from nano_inference.core.request import (
    FinishedReason,
    GenerationInputs,
    GenerationStage,
    Request,
    SamplingParams,
)
from nano_inference.driver.driver import AsyncDriver
from nano_inference.scheduler.scheduler import SimpleScheduler


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    # Step returns one token for each query
    engine.step.side_effect = lambda queries: [100] * len(queries)
    return engine


@pytest.fixture
def config():
    return SchedulerConfig(
        max_num_requests=2,
        request_timeout=0.1,  # Very short timeout for testing
        prefill_batch_delay=0.0,  # No delay in tests
    )


@pytest.mark.asyncio
async def test_admission_control(mock_engine, config):
    scheduler = SimpleScheduler()
    mock_input_processor = MagicMock()
    driver = AsyncDriver(mock_engine, scheduler, mock_input_processor, config)
    driver.start()

    try:
        req = Request(
            request_id="req1",
            generation_inputs=GenerationInputs(prompt_token_ids=[1]),
            sampling_params=SamplingParams(max_new_tokens=10),
            eos_token_id=0,
            arrival_time=time.time(),
        )

        # Add 1st request - OK
        driver.add_request(req)
        assert scheduler.get_workload() == 1

        # Add 2nd request - OK (reaches max)
        req2 = Request(
            request_id="req2",
            generation_inputs=GenerationInputs(prompt_token_ids=[1]),
            sampling_params=SamplingParams(max_new_tokens=10),
            eos_token_id=0,
            arrival_time=time.time(),
        )
        driver.add_request(req2)
        assert scheduler.get_workload() == 2

        # Add 3rd request - Should FAIL
        req3 = Request(
            request_id="req3",
            generation_inputs=GenerationInputs(prompt_token_ids=[1]),
            sampling_params=SamplingParams(max_new_tokens=10),
            eos_token_id=0,
            arrival_time=time.time(),
        )
        with pytest.raises(RuntimeError, match="Server overloaded"):
            driver.add_request(req3)

    finally:
        driver.stop()


@pytest.mark.asyncio
async def test_request_timeout(mock_engine, config):
    scheduler = SimpleScheduler()
    mock_input_processor = MagicMock()
    mock_input_processor.decode.side_effect = lambda ids, *args, **kwargs: ""
    driver = AsyncDriver(mock_engine, scheduler, mock_input_processor, config)
    driver.start()

    try:
        # Create a request that already "timed out" by arrival time
        req = Request(
            request_id="timeout-req",
            generation_inputs=GenerationInputs(prompt_token_ids=[1]),
            sampling_params=SamplingParams(max_new_tokens=10),
            eos_token_id=0,
            arrival_time=time.time() - 10.0,  # 10s ago
        )

        stream = driver.add_request(req)

        # Drain stream
        outputs = []
        async for out in stream:
            outputs.append(out)

        # The last output should have FinishedReason.ABORT
        assert len(outputs) > 0
        assert outputs[-1].finished is True
        assert outputs[-1].finished_reason == FinishedReason.ABORT

    finally:
        driver.stop()


@pytest.mark.asyncio
async def test_client_cancellation(mock_engine, config):
    scheduler = SimpleScheduler()
    mock_input_processor = MagicMock()
    mock_input_processor.decode.side_effect = lambda ids, *args, **kwargs: ""
    driver = AsyncDriver(mock_engine, scheduler, mock_input_processor, config)
    driver.start()

    try:
        req = Request(
            request_id="cancel-req",
            generation_inputs=GenerationInputs(prompt_token_ids=[1]),
            sampling_params=SamplingParams(max_new_tokens=100),  # Long generation
            eos_token_id=0,
            arrival_time=time.time(),
        )

        stream = driver.add_request(req)

        # Simulate client closing stream immediately
        async for _ in stream:
            break  # Close after first token

        # Wait a bit for the background loop and callback to complete
        await asyncio.sleep(0.1)

        # Scheduler should have 0 workload now because it was aborted
        assert scheduler.get_workload() == 0

    finally:
        driver.stop()
