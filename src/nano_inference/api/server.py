import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from nano_inference.api.protocol import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
)
from nano_inference.core.config import ModelConfig, RuntimeConfig, SchedulerConfig
from nano_inference.core.request import Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.driver import AsyncDriver
from nano_inference.engine import SingleWorkerEngine
from nano_inference.input_processor import ChatTemplateInputProcessor
from nano_inference.scheduler import OrcaScheduler
from nano_inference.utils.logger import logger
from transformers import AutoTokenizer


class AppState:
    """Holds shared application state."""

    def __init__(self, config: RuntimeConfig, inferencer_type: str = "torch"):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_dir, trust_remote_code=True
        )
        self.input_processor = ChatTemplateInputProcessor(self.tokenizer)
        self.engine = SingleWorkerEngine(inferencer_type, config.model)
        self.scheduler = OrcaScheduler(config.scheduler)
        self.driver = AsyncDriver(
            self.engine, self.scheduler, self.input_processor, config.scheduler
        )


def create_lifespan(config: RuntimeConfig, inferencer_type: str):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """FastAPI Lifespan to initialize shared state on startup."""
        app.state.state = AppState(config, inferencer_type)
        app.state.state.driver.start()
        yield
        await app.state.state.driver.stop_async()
        app.state.state = None

    return lifespan


def create_app(config: RuntimeConfig, inferencer_type: str = "torch") -> FastAPI:
    lifespan = create_lifespan(config, inferencer_type)
    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/completions")
    async def completions(
        request: CompletionRequest,
    ):
        state = app.state.state
        logger.info(f"[API] Received completion request. stream={request.stream}")

        try:
            # Convert OpenAI-style prompt to messages for chat template
            messages = [{"role": "user", "content": request.prompt}]

            # Encode via input processor
            gen_inputs = state.input_processor.encode(
                messages, add_generation_prompt=True
            )

            # Build internal Request
            arrival_time = time.time()
            request_id = f"req-{uuid.uuid4().hex}"
            logger.debug(f"[API] Assigned request_id: {request_id}")
            internal_request = Request(
                request_id=request_id,
                generation_inputs=gen_inputs,
                sampling_params=SamplingParams(
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                ),
                eos_token_id=state.tokenizer.eos_token_id,
                arrival_time=arrival_time,
            )

            if not request.stream:
                output = await state.driver.generate_async(internal_request)

                finish_reason = (
                    output.finished_reason.value if output.finished_reason else "stop"
                )
                choice = CompletionChoice(
                    text=output.delta_text,  # delta_text IS the generated text for non-streaming
                    index=0,
                    finish_reason=finish_reason,
                )

                return CompletionResponse(
                    id=request_id,
                    object="text_completion",
                    created=int(arrival_time),
                    model=request.model,
                    choices=[choice],
                    usage=None,
                )

            # Streaming: return StreamingResponse
            async def stream_generator():
                try:
                    async for output in state.driver.add_request(internal_request):
                        if not output.delta_text and not output.finished:
                            continue

                        finish_reason = None
                        if output.finished:
                            finish_reason = (
                                output.finished_reason.value
                                if output.finished_reason
                                else "stop"
                            )

                        choice = CompletionChoice(
                            text=output.delta_text,
                            index=0,
                            finish_reason=finish_reason,
                        )
                        response = CompletionResponse(
                            id=request_id,
                            object="text_completion",
                            created=int(arrival_time),
                            model=request.model,
                            choices=[choice],
                            usage=None,
                        )

                        yield f"data: {json.dumps(response.model_dump())}\n\n"
                finally:
                    pass

                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        except Exception as e:
            logger.error(f"[API] Fatal error in completions: {e}", exc_info=True)
            raise

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Any):
        # Placeholder: will implement full chat completion logic soon
        raise HTTPException(
            status_code=501, detail="Chat completions not yet implemented"
        )

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Nano-Inference API Server")
    parser.add_argument("--model-dir", help="Path to model directory")
    parser.add_argument("--device", help="Device to use (e.g., cpu, cuda)")
    parser.add_argument(
        "--dtype",
        help="Data type to use (e.g., float32, bfloat16, float16)",
    )
    parser.add_argument(
        "--inferencer-type",
        default="torch",
        help="Inferencer type to use (torch or hf)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file")

    args = parser.parse_args()

    # Unified load: YAML -> CLI Overrides -> Defaults
    runtime_config = RuntimeConfig.load(yaml_path=args.config, cli_overrides=vars(args))

    app = create_app(runtime_config, inferencer_type=args.inferencer_type)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )
