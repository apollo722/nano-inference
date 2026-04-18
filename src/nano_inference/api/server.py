import argparse
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from nano_inference.api.protocol import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
)
from nano_inference.core.config import ModelConfig
from nano_inference.core.request import GenerationInputs, Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.driver import SyncDriver
from nano_inference.engine import SingleWorkerEngine
from nano_inference.input_processor import ChatTemplateInputProcessor
from transformers import AutoTokenizer


class AppState:
    """Holds shared application state."""

    def __init__(self, config: ModelConfig, inferencer_type: str = "torch"):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_dir, trust_remote_code=True
        )
        self.input_processor = ChatTemplateInputProcessor(self.tokenizer)
        self.engine = SingleWorkerEngine(inferencer_type, config)
        self.driver = SyncDriver(self.engine)


def create_lifespan(config: ModelConfig, inferencer_type: str):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """FastAPI Lifespan to initialize shared state on startup."""
        app.state.state = AppState(config, inferencer_type)
        yield
        app.state.state = None

    return lifespan


def create_app(config: ModelConfig, inferencer_type: str = "torch") -> FastAPI:
    lifespan = create_lifespan(config, inferencer_type)
    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(request: CompletionRequest) -> CompletionResponse:
        state = app.state.state

        # Convert OpenAI-style prompt to messages for chat template
        messages = [{"role": "user", "content": request.prompt}]

        # Encode via input processor
        gen_inputs = state.input_processor.encode(messages, add_generation_prompt=True)

        # Build internal Request
        internal_request = Request(
            request_id=f"req-{int(time.time() * 1000)}",
            generation_inputs=gen_inputs,
            sampling_params=SamplingParams(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            ),
            eos_token_id=state.tokenizer.eos_token_id,
            arrival_time=time.time(),
        )

        # Run through driver
        output = state.driver.generate(internal_request)

        # Decode output tokens
        generated_text = state.input_processor.decode(output.output_token_ids)

        # Build response
        finish_reason = (
            output.finished_reason.value if output.finished_reason else "stop"
        )
        choice = CompletionChoice(
            text=generated_text,
            index=0,
            finish_reason=finish_reason,
        )

        return CompletionResponse(
            id=internal_request.request_id,
            object="text_completion",
            created=int(internal_request.arrival_time),
            model=request.model,
            choices=[choice],
            usage=None,
        )

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Nano-Inference API Server")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument(
        "--device", default="cpu", help="Device to use (e.g., cpu, cuda)"
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Data type to use (e.g., float32, bfloat16, float16)",
    )
    parser.add_argument(
        "--inferencer-type",
        default="torch",
        help="Inferencer type to use (torch or hf)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    args = parser.parse_args()

    config = ModelConfig(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )

    app = create_app(config, inferencer_type=args.inferencer_type)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )
