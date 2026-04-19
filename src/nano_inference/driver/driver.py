import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, List

from nano_inference.core.config import ModelConfig, SchedulerConfig
from nano_inference.core.request import (
    FinishedReason,
    GenerateOutput,
    GenerateQuery,
    GenerationStage,
    Request,
)
from nano_inference.driver.output_processor import OutputProcessor
from nano_inference.driver.response_manager import ResponseManager
from nano_inference.engine.engine import EngineBase
from nano_inference.scheduler.scheduler import SchedulerBase
from nano_inference.utils.logger import logger


class DriverBase(ABC):
    """Abstract base for the driver - the public entry point of the serving stack.

    The driver owns the request lifecycle:
    1. Accept a Request (external, immutable)
    2. Convert to GenerateQuery (internal, mutable)
    3. Delegate to Engine
    4. Return GenerateOutput

    Phase 1: SyncDriver (blocking, single request).
    Phase 2: AsyncDriver (non-blocking, concurrent requests, scheduler integration).
    """

    @abstractmethod
    def generate(self, request: Request) -> GenerateOutput:
        """Process a single request end-to-end."""
        ...


class SyncDriver(DriverBase):
    """Synchronous driver: blocks until generation is complete.

    Phase 3: Now uses the AsyncDriver infrastructure internally to ensure
    consistent KV block allocation and paged attention.
    """

    def __init__(
        self,
        engine: EngineBase,
        scheduler: SchedulerBase,
        input_processor: Any,
        config: SchedulerConfig = None,
    ):
        self.async_driver = AsyncDriver(engine, scheduler, input_processor, config)

    def generate(self, request: Request) -> GenerateOutput:
        self.async_driver.start()
        try:
            return self.async_driver.generate(request)
        finally:
            self.async_driver.stop()


class AsyncDriver(DriverBase):
    """Asynchronous driver: non-blocking, with background scheduler loop.

    - Public API: `generate(request)` (blocks for backward compatibility)
    - Non-blocking API: `add_request(request)` (returns Future[GenerateOutput])
    - Background thread runs scheduling loop
    """

    def __init__(
        self,
        engine: EngineBase,
        scheduler: SchedulerBase,
        input_processor: Any,
        config: SchedulerConfig = None,
    ):
        self.engine = engine
        self.scheduler = scheduler
        self.config = config
        self.response_manager = ResponseManager()
        self.output_processor = OutputProcessor(input_processor)

        self._running = False
        self._generate_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_stats_log_time = 0.0

    async def generate_async(self, request: Request) -> GenerateOutput:
        """Asynchronous API: wait for and return the final output."""
        last_output = None
        async for output in self.add_request(request):
            last_output = output

        if last_output is None:
            raise RuntimeError("Request failed or was aborted before generating output")
        return last_output

    def generate(self, request: Request) -> GenerateOutput:
        """Blocking API (for backward compatibility). Assert not in event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop in this thread, safe to use asyncio.run
            return asyncio.run(self.generate_async(request))

        if loop.is_running():
            # We are inside a running event loop (e.g. FastAPI thread).
            # Blocking here with a sync call would deadlock the loop.
            raise RuntimeError(
                "AsyncDriver.generate() is a blocking call and cannot be used "
                "inside a running event loop. Use 'await driver.generate_async(request)' instead."
            )

        # Loop exists but is not running (rare, but possible in some setups)
        return asyncio.run(self.generate_async(request))

    def add_request(self, request: Request) -> AsyncGenerator[GenerateOutput, None]:
        """Non-blocking API: add request and return an AsyncGenerator of outputs."""
        with self._lock:
            if not self._running:
                raise RuntimeError("AsyncDriver not running - call start() first")

            # Admission Control (Phase 2.7)
            if (
                self.config
                and self.scheduler.get_workload() >= self.config.max_num_requests
            ):
                raise RuntimeError(
                    f"Server overloaded. Max concurrent requests ({self.config.max_num_requests}) reached."
                )

            # Define cancellation callback
            def on_cancel():
                logger.info(f"Request {request.request_id} cancelled by client.")
                self.scheduler.abort_tasks({request.request_id})

            gen = self.response_manager.create_stream(
                request.request_id, on_close=on_cancel
            )
            query = GenerateQuery.from_request(request)
            self.scheduler.add_tasks([query])

        return gen

    def start(self) -> None:
        """Start the background scheduling loop."""
        with self._lock:
            if self._running:
                return
            logger.info("Starting AsyncDriver background thread...")
            self._running = True
            self._generate_thread = threading.Thread(
                target=self._generate_loop, daemon=True
            )
            self._generate_thread.start()

    async def stop_async(self) -> None:
        """Stop the background scheduling loop asynchronously."""
        logger.info("Stopping AsyncDriver (async)...")
        with self._lock:
            if not self._running:
                logger.info("AsyncDriver already stopped.")
                return
            self._running = False

        # 1. Signal all active requests to close immediately.
        self.response_manager.shutdown()

        # 2. Abort all tasks in scheduler
        self.scheduler.abort_tasks(set())

        # 3. Wait for the background thread to finish its current step.
        if self._generate_thread:
            logger.info("Waiting for AsyncDriver background thread to exit...")
            # Use asyncio.to_thread to join the thread without blocking the loop
            try:
                await asyncio.to_thread(self._generate_thread.join, 10.0)
            except Exception as e:
                logger.error(f"Error while joining AsyncDriver thread: {e}")

            if self._generate_thread.is_alive():
                logger.warning("AsyncDriver thread did not exit within timeout.")
            else:
                logger.info("AsyncDriver thread exited cleanly.")
        else:
            logger.info("No background thread found to stop.")

    def stop(self) -> None:
        """Stop the background scheduling loop synchronously (for non-async callers)."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        self.response_manager.shutdown()
        if self._generate_thread:
            self._generate_thread.join(timeout=10.0)

    def _generate_loop(self) -> None:
        """Background loop: schedule queries, run engine, deliver responses."""
        logger.info("AsyncDriver generation loop started.")
        step_count = 0
        while True:
            try:
                # 1. Check if we should keep running (minimal lock duration)
                with self._lock:
                    if not self._running:
                        logger.info("AsyncDriver loop stopping flag detected.")
                        break

                # 2. Schedule and execute (NO LOCK HELD)
                queries = self.scheduler.schedule()
                if not queries:
                    time.sleep(0.01)
                    continue

                step_count += 1
                batch_size = len(queries)
                logger.debug(
                    f"[Driver] Step {step_count} starting for batch of {batch_size}"
                )
                start_time = time.time()

                try:
                    # Step the engine for the current batch
                    new_token_ids = self.engine.step(queries)

                    # Process outputs and transition states
                    self.output_processor.process_step_outputs(queries, new_token_ids)
                except Exception as e:
                    logger.error(
                        f"[Driver] Error during inference step {step_count}: {e}",
                        exc_info=True,
                    )
                    for query in queries:
                        self.response_manager.fail(query.request_id, e)
                        self.scheduler.finish_tasks([query.request_id])
                    continue

                elapsed = time.time() - start_time
                logger.debug(f"[Driver] Step {step_count} completed in {elapsed:.3f}s")

                # Handle results: finish or reschedule
                reschedule_queries = []

                for query in queries:
                    # 1. Timeout Handling (Phase 2.7)
                    if (
                        self.config
                        and (time.time() - query.arrival_time)
                        > self.config.request_timeout
                    ):
                        query.stage = GenerationStage.FINISHED
                        query.finished_reason = FinishedReason.ABORT
                        logger.warning(
                            f"[Driver] Request {query.request_id} TIMED OUT."
                        )

                    # 2. Check if request was cancelled while we were processing the step
                    if not self.response_manager.has_request(query.request_id):
                        logger.debug(
                            f"[Driver] Request {query.request_id} was CANCELLED during step."
                        )
                        self.scheduler.finish_tasks([query.request_id])
                        continue

                    output = GenerateOutput(
                        output_token_ids=query.output_token_ids,
                        finished=(query.stage == GenerationStage.FINISHED),
                        finished_reason=getattr(query, "finished_reason", None),
                        delta_text=getattr(query, "delta_text", ""),
                        full_text=getattr(query, "full_text", ""),
                    )

                    logger.debug(
                        f"[Driver] Delivering output for {query.request_id} (finished={output.finished}, delta_len={len(output.delta_text)})"
                    )
                    self.response_manager.put_output(query.request_id, output)

                    if query.stage == GenerationStage.FINISHED:
                        logger.info(
                            f"[Driver] Request {query.request_id} FINISHED (reason: {output.finished_reason})"
                        )
                        self.response_manager.complete(query.request_id)
                        # Phase 3: Explicitly finish in scheduler to free KV blocks
                        self.scheduler.finish_tasks([query.request_id])
                    else:
                        reschedule_queries.append(query)

                # Always release 'running' status in the scheduler before rescheduling
                # This doesn't free the KV cache, just allows the task to be scheduled again.
                self.scheduler.on_step_completed([q.request_id for q in queries])

                if reschedule_queries:
                    logger.debug(
                        f"[Driver] Rescheduling {len(reschedule_queries)} queries"
                    )
                    self.scheduler.add_tasks(reschedule_queries)

                # 3. Log metrics periodically (Phase 3 deliverable: KV utilization visible)
                current_time = time.time()
                if current_time - self._last_stats_log_time >= 5.0:
                    stats = self.scheduler.get_stats()
                    running = stats.get("num_running", 0)
                    prefill = stats.get("num_prefill_waiting", 0)
                    decode = stats.get("num_decode_waiting", 0)
                    kv_util = stats.get("kv_utilization", 0.0) * 100

                    logger.info(
                        f"[Metrics] Running={running} | Waiting(P/D)={prefill}/{decode} | "
                        f"KV Util={kv_util:.1f}%"
                    )
                    self._last_stats_log_time = current_time

            except Exception as e:
                logger.error(
                    f"[Driver] Fatal error in AsyncDriver loop: {e}", exc_info=True
                )
                time.sleep(0.1)
        logger.info("AsyncDriver generation loop exited.")
