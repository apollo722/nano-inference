from __future__ import annotations

import argparse
import json

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a request to the Nano-Inference API server.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port.",
    )
    parser.add_argument(
        "--model",
        default="qwen3-0.6b",
        help="Model name to use.",
    )
    parser.add_argument(
        "--prompt",
        default="Count: one, two, three,",
        help="Prompt text to generate from.",
    )
    parser.add_argument(
        "--max-tokens",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    url = f"http://{args.host}:{args.port}/v1/completions"

    print(f"Sending request to {url}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print()

    response = requests.post(
        url,
        json={
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    )

    response.raise_for_status()
    data = response.json()

    print("=== Response ===")
    print(json.dumps(data, indent=2))
    print()

    if data.get("choices"):
        choice = data["choices"][0]
        print("=== Generated Text ===")
        print(choice["text"])
        print()
        print("=== Finish Reason ===")
        print(choice.get("finish_reason", "unknown"))


if __name__ == "__main__":
    main()
