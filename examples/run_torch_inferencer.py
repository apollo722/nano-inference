from __future__ import annotations

import argparse
import time
import uuid

import torch
from nano_inference.core.config import ModelConfig
from nano_inference.core.request import Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.inferencer.torch_inferencer import TorchInferencer
from nano_inference.input_processor import ChatTemplateInputProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local TorchInferencer against a local model path.",
    )
    parser.add_argument("--model-dir", required=True, help="Local model directory.")
    parser.add_argument(
        "--prompt",
        default="Who are you?",
        help="Prompt text to tokenize and generate from.",
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

    inferencer = TorchInferencer()
    inferencer.load_model(
        ModelConfig(
            model_dir=args.model_dir,
            device=args.device,
            dtype=args.dtype,
        )
    )

    # Use the input processor for text-only models
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
    print()

    assembled_text = ""
    for _, token_id in enumerate(output.output_token_ids, start=1):
        token_text = inferencer.tokenizer.decode([token_id], skip_special_tokens=False)
        assembled_text += token_text

    print("=== Final Text ===")
    print(assembled_text)
    print()
    print("=== Finished Reason ===")
    print(
        output.finished_reason.value
        if output.finished_reason is not None
        else "unknown"
    )


if __name__ == "__main__":
    main()
