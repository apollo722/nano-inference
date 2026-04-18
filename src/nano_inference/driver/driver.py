import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future

from nano_inference.core.request import GenerateOutput, GenerateQuery, Request
from nano_inference.driver.response_manager import ResponseManager
from nano_inference.engine.engine import EngineBase
from nano_inference.scheduler.scheduler import SchedulerBase


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

    Converts Request -> GenerateQuery at the boundary, then delegates to Engine.
    Phase 2 will replace this with AsyncDriver that adds a scheduler loop.
    """

    def __init__(self, engine: EngineBase):
        self.engine = engine

    def generate(self, request: Request) -> GenerateOutput:
        query = GenerateQuery.from_request(request)
        outputs = self.engine.generate([query])
        return outputs[0]


class AsyncDriver(DriverBase):
    """Asynchronous driver: non-blocking, with background scheduler loop.

    - Public API: `generate(request)` (blocks for backward compatibility)
    - Non-blocking API: `add_request(request)` (returns Future[GenerateOutput])
    - Background thread runs scheduling loop
    """

    def __init__(self, engine: EngineBase, scheduler: SchedulerBase):
        self.engine = engine
        self.scheduler = scheduler
        self.response_manager = ResponseManager()

        self._running = False
        self._generate_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def generate(self, request: Request) -> GenerateOutput:
        """Blocking API (for backward compatibility with SyncDriver)."""
        future = self.add_request(request)
        return future.result()

    def add_request(self, request: Request) -> Future[GenerateOutput]:
        """Non-blocking API: add request and return Future."""
        with self._lock:
            if not self._running:
                raise RuntimeError("AsyncDriver not running - call start() first")

            future = self.response_manager.create_future(request.request_id)
            query = GenerateQuery.from_request(request)
            self.scheduler.add_tasks([query])

        return future

    def start(self) -> None:
        """Start the background scheduling loop."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._generate_thread = threading.Thread(
                target=self._generate_loop, daemon=True
            )
            self._generate_thread.start()

    def stop(self) -> None:
        """Stop the background scheduling loop."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._generate_thread:
            self._generate_thread.join(timeout=5.0)

    def _generate_loop(self) -> None:
        """Background loop: schedule queries, run engine, deliver responses."""
        while True:
            with self._lock:
                if not self._running:
                    break

            queries = self.scheduler.schedule()
            if not queries:
                time.sleep(0.001)
                continue

            outputs = self.engine.generate(queries)
            assert len(queries) == len(outputs)

            request_ids_to_finish = []
            for query, output in zip(queries, outputs):
                request_ids_to_finish.append(query.request_id)
                if output.finished:
                    self.response_manager.complete(query.request_id, output)
                else:
                    self.scheduler.add_tasks([query])

                self.scheduler.finish_tasks(request_ids_to_finish)
