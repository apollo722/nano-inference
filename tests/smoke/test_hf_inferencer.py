import os
import time
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import pytest
from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerationInputs, Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.inferencer.factory import InferencerFactory
from PIL import Image, ImageDraw

from tests.utils import ensure_test_model_downloaded


def _make_test_image(path: Path) -> None:
    img = Image.new("RGB", (224, 224), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 200, 200), outline="blue", width=6)
    draw.ellipse((70, 70, 150, 150), fill="red")
    draw.text((40, 180), "red circle", fill="black")
    img.save(path)


@pytest.mark.smoke
def test_hf_text_only_generate_greedy_4_tokens():
    model_path = ensure_test_model_downloaded("Qwen/Qwen3-0.6B")
    inf = InferencerFactory.create(
        "huggingface", ModelConfig(model_dir=model_path, device="cpu", dtype="float32")
    )

    prompt = "Count: one, two, three,"
    token_ids = inf.tokenizer.encode(prompt)
    generation_inputs = GenerationInputs(prompt_token_ids=token_ids)
    req = Request(
        request_id="smoke-text",
        generation_inputs=generation_inputs,
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        eos_token_id=inf.tokenizer.eos_token_id,
        arrival_time=time.time(),
    )

    out = inf.generate(req)
    assert len(out.output_token_ids) == 4

    decoded = inf.tokenizer.decode(out.output_token_ids, skip_special_tokens=True)
    assert isinstance(decoded, str) and len(decoded) > 0


@pytest.mark.smoke
def test_hf_vlm_generate_from_messages_greedy_4_tokens(tmp_path: Path):
    model_path = ensure_test_model_downloaded("Qwen/Qwen2.5-VL-3B-Instruct")
    inf = InferencerFactory.create(
        "huggingface", ModelConfig(model_dir=model_path, device="cpu", dtype="float32")
    )

    assert inf.is_vlm, "Model should be detected as VLM"

    img_path = tmp_path / "test_image.png"
    _make_test_image(img_path)

    # For VL, use apply_chat_template to get the text that includes image placeholders
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": "Describe the image in a few words."},
            ],
        }
    ]

    # Use tokenizer's (via processor) chat template to get the right text with image plac
    text = inf.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize that text to get prompt_token_ids, no images in this tokenization call!
    token_ids = inf.tokenizer.encode(text)

    image = Image.open(img_path)
    generation_inputs = GenerationInputs(
        prompt_token_ids=token_ids,
        images=[image],
    )

    req = Request(
        request_id="smoke-vlm",
        generation_inputs=generation_inputs,
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        eos_token_id=inf.tokenizer.eos_token_id,
        arrival_time=time.time(),
    )

    out = inf.generate(req)

    # We don't assert exact text; just ensure we got a short, non-empty string.
    decoded = inf.tokenizer.decode(out.output_token_ids, skip_special_tokens=True)
    assert isinstance(decoded, str)
    assert len(decoded.strip()) > 0

    # Approximate new token count >= 1
    approx_tokens = inf.tokenizer.encode(decoded, add_special_tokens=False)
    assert len(approx_tokens) >= 1
