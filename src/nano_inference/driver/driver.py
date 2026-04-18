from abc import ABC, abstractmethod

from nano_inference.core.request import GenerateOutput, GenerateQuery, Request
from nano_inference.engine.engine import EngineBase


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
