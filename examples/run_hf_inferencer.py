from __future__ import annotations

import argparse
import os
import time
import uuid

import torch

# Suppress transformers warnings about invalid generation flags
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerationInputs, Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.inferencer.hf_inferencer import HuggingFaceInferencer
from nano_inference.input_processor import ChatTemplateInputProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local HuggingFaceInferencer against a local model path.",
    )
    parser.add_argument("--model-dir", required=True, help="Local model directory.")
    parser.add_argument(
        "--prompt",
        default="Who are you?",
        help="Prompt text to tokenize and generate from.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image path or URL (for VL models).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling cutoff. Use -1 to disable.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling cutoff.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Data type to use.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    inferencer = HuggingFaceInferencer()
    inferencer.load_model(
        ModelConfig(
            model_dir=args.model_dir,
            device=args.device,
            dtype=args.dtype,
        )
    )

    # Build GenerationInputs
    if inferencer.is_vlm:
        from qwen_vl_utils import process_vision_info

        # Build messages with image if provided
        if args.image:
            content = [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ]
        else:
            content = args.prompt

        messages = [{"role": "user", "content": content}]

        # Apply chat template to get text
        text = inferencer.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Get prompt_token_ids by tokenizing the text
        model_inputs = inferencer.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        prompt_token_ids = model_inputs["input_ids"][0].tolist()

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Create GenerationInputs with vision inputs
        gen_inputs = GenerationInputs(
            prompt_token_ids=prompt_token_ids,
            images=image_inputs,
            videos=video_inputs,
        )

    else:
        # For text-only models, use the input processor
        processor = ChatTemplateInputProcessor(tokenizer=inferencer.tokenizer)
        messages = [{"role": "user", "content": args.prompt}]
        gen_inputs = processor.encode(messages)

    request = Request(
        request_id=f"example-{uuid.uuid4().hex[:8]}",
        generation_inputs=gen_inputs,
        sampling_params=SamplingParams(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
        eos_token_id=inferencer.tokenizer.eos_token_id,
        arrival_time=time.time(),
    )

    output = inferencer.generate(request)

    print("=== Prompt ===")
    print(args.prompt)
    if args.image:
        print(f"Image: {args.image}")
    print()

    print("=== Final Text ===")
    if inferencer.is_vlm:
        print(
            inferencer.processor.decode(
                output.output_token_ids, skip_special_tokens=True
            )
        )
    else:
        print(
            inferencer.tokenizer.decode(
                output.output_token_ids, skip_special_tokens=True
            )
        )
    print()

    print("=== Finished Reason ===")
    print(
        output.finished_reason.value
        if output.finished_reason is not None
        else "unknown"
    )


if __name__ == "__main__":
    main()
