import time

import pytest
from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerationInputs, Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.inferencer.factory import InferencerFactory

from tests.utils import ensure_test_model_downloaded


def _build_request(inferencer, prompt: str) -> Request:
    token_ids = inferencer.tokenizer.encode(prompt)
    generation_inputs = GenerationInputs(prompt_token_ids=token_ids)
    return Request(
        request_id="smoke-torch-text",
        generation_inputs=generation_inputs,
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        eos_token_id=inferencer.tokenizer.eos_token_id,
        arrival_time=time.time(),
    )


@pytest.mark.smoke
def test_torch_text_only_generate_greedy_4_tokens():
    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    torch_inferencer = InferencerFactory.create(
        "torch", ModelConfig(model_dir=model_path, device="cpu", dtype="float32")
    )

    request = _build_request(torch_inferencer, "Count: one, two, three,")
    torch_output = torch_inferencer.generate(request)
    torch_text = torch_inferencer.tokenizer.decode(
        torch_output.output_token_ids, skip_special_tokens=True
    )

    assert len(torch_output.output_token_ids) == 4
    assert isinstance(torch_text, str) and len(torch_text) > 0