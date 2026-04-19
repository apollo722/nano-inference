from __future__ import annotations

import argparse

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Nano-Inference API server.",
    )
    parser.add_argument("--model-dir", help="Local model directory.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Data type to use (e.g., float32, bfloat16, float16).",
    )
    parser.add_argument(
        "--inferencer-type",
        default="torch",
        choices=["torch", "huggingface"],
        help="Inferencer type to use.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to listen on.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--max-prefill-batch-size",
        type=int,
        help="Maximum number of requests to prefill in one batch.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Import here to avoid circular imports and heavy dependencies before argparse
    import uvicorn
    from nano_inference.api.server import create_app
    from nano_inference.core.config import RuntimeConfig

    # Unified load: YAML -> CLI Overrides -> Defaults
    runtime_config = RuntimeConfig.load(yaml_path=args.config, cli_overrides=vars(args))

    app = create_app(runtime_config, inferencer_type=args.inferencer_type)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
