import time

import pytest
from nano_inference.core.config import ModelConfig, RuntimeConfig
from nano_inference.core.request import GenerationInputs, Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.driver.driver import SyncDriver
from nano_inference.engine.engine import SingleWorkerEngine

from tests.utils import ensure_test_model_downloaded


def _make_driver(model_path: str) -> SyncDriver:
    model_config = ModelConfig(model_dir=model_path, device="cpu", dtype="float32")
    engine = SingleWorkerEngine(inferencer_type="torch", model_config=model_config)
    return SyncDriver(engine=engine)


def _build_request(tokenizer, prompt: str, request_id: str = "smoke-driver") -> Request:
    token_ids = tokenizer.encode(prompt)
    return Request(
        request_id=request_id,
        generation_inputs=GenerationInputs(prompt_token_ids=token_ids),
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        eos_token_id=tokenizer.eos_token_id,
        arrival_time=time.time(),
    )


@pytest.mark.smoke
def test_driver_stack_generates_4_tokens():
    """Full stack: SyncDriver -> SingleWorkerEngine -> Worker -> TorchInferencer."""
    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    driver = _make_driver(model_path)

    # Access tokenizer through the stack for building the request
    tokenizer = driver.engine.worker.inferencer.tokenizer

    request = _build_request(tokenizer, "Count: one, two, three,")
    output = driver.generate(request)
    text = tokenizer.decode(output.output_token_ids, skip_special_tokens=True)

    assert len(output.output_token_ids) == 4
    assert isinstance(text, str) and len(text) > 0


@pytest.mark.smoke
def test_driver_output_matches_direct_inferencer():
    """Driver stack should produce identical output to calling TorchInferencer directly."""
    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    driver = _make_driver(model_path)

    tokenizer = driver.engine.worker.inferencer.tokenizer
    inferencer = driver.engine.worker.inferencer

    prompt = "Count: one, two, three,"
    request = _build_request(tokenizer, prompt)

    # Through the full stack
    stack_output = driver.generate(request)

    # Direct inferencer call
    direct_request = _build_request(tokenizer, prompt, request_id="smoke-direct")
    direct_output = inferencer.generate(direct_request)

    assert stack_output.output_token_ids == direct_output.output_token_ids
    assert stack_output.finished == direct_output.finished
    assert stack_output.finished_reason == direct_output.finished_reason
