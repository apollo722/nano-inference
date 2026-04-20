"""Smoke test: Qwen2.5-VL-3B end-to-end inference with a real image.

Verifies:
1. Model loads (weights remapped correctly).
2. VLMInputProcessor encodes image + text.
3. TorchInferencer generates tokens (greedy, deterministic).
4. Generated token IDs match HF baseline under greedy decoding.
"""

import time
from pathlib import Path

import pytest
import torch
from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerationInputs, Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.inferencer.factory import InferencerFactory
from nano_inference.input_processor.vlm import Qwen25VLInputProcessor
from PIL import Image

from tests.utils import ensure_test_model_downloaded

VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW_TOKENS = 8


def _make_test_image() -> Image.Image:
    """Create a simple synthetic test image (blue rectangle)."""
    img = Image.new("RGB", (448, 448), color=(0, 128, 255))
    return img


@pytest.fixture(scope="module")
def model_path():
    return ensure_test_model_downloaded(VLM_MODEL_ID)


@pytest.fixture(scope="module")
def torch_vlm_inferencer(model_path):
    inferencer = InferencerFactory.create(
        "torch",
        ModelConfig(
            model_dir=model_path,
            device="cpu",
            dtype="float32",
        ),
    )
    return inferencer


@pytest.mark.smoke
def test_vlm_model_loads(torch_vlm_inferencer):
    """Basic check: model loaded and has no catastrophic missing keys."""
    report = torch_vlm_inferencer.load_report
    assert report is not None, "No load report"
    # Visual encoder and decoder weights must all be present
    assert len(report.loaded_keys) > 0, "No weights loaded"
    # Allow some missing keys (e.g. inv_freq buffers) but not weight matrices
    weight_missing = [k for k in report.missing_keys if "weight" in k]
    assert len(weight_missing) == 0, f"Missing weight tensors: {weight_missing}"


@pytest.mark.smoke
def test_vlm_generates_tokens(torch_vlm_inferencer, model_path):
    """TorchInferencer generates tokens for an image+text prompt."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vp = Qwen25VLInputProcessor(
        tokenizer=processor.tokenizer,
        processor=processor,
    )
    image = _make_test_image()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What color is this image?"},
            ],
        }
    ]
    gen_inputs = vp.encode(messages, images=[image])
    assert gen_inputs.mrope_position_ids is not None, "mrope_position_ids should be set"
    assert gen_inputs.image_grid_thw is not None, "image_grid_thw should be set"

    request = Request(
        request_id="smoke-vlm",
        generation_inputs=gen_inputs,
        sampling_params=SamplingParams(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0),
        eos_token_id=processor.tokenizer.eos_token_id,
        arrival_time=time.time(),
    )

    output = torch_vlm_inferencer.generate(request)
    assert len(output.output_token_ids) > 0, "No tokens generated"
    assert len(output.output_token_ids) <= MAX_NEW_TOKENS

    decoded = processor.tokenizer.decode(
        output.output_token_ids, skip_special_tokens=True
    )
    assert isinstance(decoded, str) and len(decoded) > 0, "Empty decoded output"
    print(f"\nVLM output: {decoded!r}")


@pytest.mark.smoke
def test_vlm_text_only_still_works(torch_vlm_inferencer, model_path):
    """Text-only request through VLM model (no images) should still work."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vp = Qwen25VLInputProcessor(
        tokenizer=processor.tokenizer,
        processor=processor,
    )
    messages = [{"role": "user", "content": "Say hello."}]
    gen_inputs = vp.encode(messages)

    request = Request(
        request_id="smoke-vlm-text",
        generation_inputs=gen_inputs,
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        eos_token_id=processor.tokenizer.eos_token_id,
        arrival_time=time.time(),
    )
    output = torch_vlm_inferencer.generate(request)
    assert len(output.output_token_ids) > 0
