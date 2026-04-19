from __future__ import annotations

import argparse
import json

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a chat request to the Nano-Inference API server.",
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
        "--message",
        default="Who are you?",
        help="The message to send to the assistant.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Whether to stream the response.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    print(f"Sending chat request to {url}")
    print(f"Model: {args.model}")
    print(f"Message: {args.message}")
    print(f"Stream: {args.stream}")
    print()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.message}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
    }

    if not args.stream:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        print("=== Response ===")
        print(json.dumps(data, indent=2))
        print()

        if data.get("choices"):
            choice = data["choices"][0]
            if "message" in choice:
                print("=== Assistant Message ===")
                print(choice["message"]["content"])

        if data.get("usage"):
            print("\n=== Usage ===")
            print(json.dumps(data["usage"], indent=2))
    else:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        print("=== Assistant (Streaming) ===")
        last_data = None
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[len("data: ") :]
                if data_str == "[DONE]":
                    break
                data = json.loads(data_str)
                last_data = data
                if data.get("choices"):
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        print(delta["content"], end="", flush=True)
        print("\n\n=== Stream Finished ===")
        if last_data and last_data.get("usage"):
            print("\n=== Final Usage ===")
            print(json.dumps(last_data["usage"], indent=2))


if __name__ == "__main__":
    main()
