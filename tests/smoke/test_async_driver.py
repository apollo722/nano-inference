import time

import pytest
from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerationInputs, Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.driver.driver import AsyncDriver
from nano_inference.engine.engine import SingleWorkerEngine
from nano_inference.inferencer.factory import InferencerFactory
from nano_inference.scheduler.scheduler import SimpleScheduler

from tests.utils import ensure_test_model_downloaded


def _build_request(inferencer, prompt: str) -> Request:
    token_ids = inferencer.tokenizer.encode(prompt)
    generation_inputs = GenerationInputs(prompt_token_ids=token_ids)
    return Request(
        request_id="smoke-async-text",
        generation_inputs=generation_inputs,
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        eos_token_id=inferencer.tokenizer.eos_token_id,
        arrival_time=time.time(),
    )


@pytest.mark.smoke
def test_async_driver_generates_4_tokens():
    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    config = ModelConfig(model_dir=model_path, device="cpu", dtype="float32")

    # Create a dummy inferencer just to get tokenizer for request building
    dummy_inferencer = InferencerFactory.create("torch", config)

    engine = SingleWorkerEngine(inferencer_type="torch", model_config=config)
    scheduler = SimpleScheduler()
    driver = AsyncDriver(engine=engine, scheduler=scheduler)

    driver.start()
    try:
        request = _build_request(dummy_inferencer, "Count: one, two, three,")
        output = driver.generate(request)

        assert len(output.output_token_ids) == 4
    finally:
        driver.stop()
