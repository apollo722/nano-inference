import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from nano_inference.api.protocol import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
)
from nano_inference.core.config import ModelConfig, RuntimeConfig, SchedulerConfig
from nano_inference.core.request import GenerationInputs, Request
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
        self.driver = AsyncDriver(self.engine, self.scheduler, config.scheduler)


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
                # Non-streaming: drain the generator asynchronously
                output = None
                async for chunk in state.driver.add_request(internal_request):
                    output = chunk

                if output is None:
                    raise RuntimeError("No output generated from driver")

                # Decode output tokens
                generated_text = state.input_processor.decode(output.output_token_ids)
                logger.info(f"[API] Request {request_id} completed successfully.")

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
                    id=request_id,
                    object="text_completion",
                    created=int(arrival_time),
                    model=request.model,
                    choices=[choice],
                    usage=None,
                )

            # Streaming: return StreamingResponse
            async def stream_generator():
                logger.debug(f"[API] Stream generator started for {request_id}")
                previous_tokens_len = 0
                try:
                    async for output in state.driver.add_request(internal_request):
                        # Decode only the NEW tokens
                        new_tokens = output.output_token_ids[previous_tokens_len:]
                        if not new_tokens and not output.finished:
                            continue

                        delta_text = state.input_processor.decode(new_tokens)
                        previous_tokens_len = len(output.output_token_ids)

                        finish_reason = None
                        if output.finished:
                            finish_reason = (
                                output.finished_reason.value
                                if output.finished_reason
                                else "stop"
                            )

                        choice = CompletionChoice(
                            text=delta_text,
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
                except Exception as e:
                    logger.error(
                        f"[API] Error in stream_generator for {request_id}: {e}"
                    )
                    raise
                finally:
                    logger.debug(f"[API] Stream generator finished for {request_id}")

                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        except Exception as e:
            logger.error(f"[API] Fatal error in completions: {e}", exc_info=True)
            raise

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
