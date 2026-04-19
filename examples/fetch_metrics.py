from __future__ import annotations

import argparse
import json

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch metrics from the Nano-Inference API server.",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    url = f"http://{args.host}:{args.port}/metrics"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        print("=== Server Metrics ===")
        print(json.dumps(data, indent=2))

        if "kv_utilization" in data:
            util = data["kv_utilization"] * 100
            print(f"\nKV Cache Utilization: {util:.1f}%")

    except Exception as e:
        print(f"Error fetching metrics: {e}")


if __name__ == "__main__":
    main()
