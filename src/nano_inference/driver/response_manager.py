import asyncio
from typing import AsyncGenerator, Dict, Optional

from nano_inference.core.request import GenerateOutput
from nano_inference.utils.logger import logger


class ResponseManager:
    def __init__(self):
        # Maps request_id to [asyncio.Queue, asyncio.AbstractEventLoop, finished_flag]
        self._queues: Dict[str, list] = {}
        self._is_shutting_down = False

    def create_stream(
        self, request_id: str, on_close: Optional[callable] = None
    ) -> AsyncGenerator[GenerateOutput, None]:
        """Create a new output queue and return an async generator."""
        if self._is_shutting_down:
            raise RuntimeError("ResponseManager is shutting down.")

        logger.debug(f"[ResponseManager] Creating stream for {request_id}")
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        # Using a list [queue, loop, is_finished] to allow in-place modification of the flag
        state = [queue, loop, False]
        self._queues[request_id] = state

        async def generator():
            logger.debug(f"[ResponseManager] Generator started for {request_id}")
            try:
                while True:
                    output = await queue.get()
                    if output is None:  # End of stream marker
                        logger.debug(
                            f"[ResponseManager] Generator received None (completion) for {request_id}"
                        )
                        state[2] = True  # Mark as finished naturally
                        break
                    yield output
            except Exception as e:
                logger.error(
                    f"[ResponseManager] Error in generator for {request_id}: {e}"
                )
                raise
            finally:
                logger.debug(
                    f"[ResponseManager] Generator finally block for {request_id}"
                )
                # Cleanup if generator is closed
                if request_id in self._queues:
                    _, _, finished_naturally = self._queues.pop(request_id)
                    logger.debug(
                        f"[ResponseManager] Popped {request_id}, finished_naturally={finished_naturally}"
                    )
                    # Only call on_close if the stream was NOT finished naturally
                    if not finished_naturally and on_close:
                        logger.info(
                            f"[ResponseManager] Triggering on_close for {request_id}"
                        )
                        on_close()

        return generator()

    def put_output(self, request_id: str, output: GenerateOutput) -> None:
        """Thread-safe put output into the queue."""
        if request_id not in self._queues:
            logger.debug(
                f"[ResponseManager] put_output failed: {request_id} not in queues"
            )
            return
        queue, loop, _ = self._queues[request_id]
        loop.call_soon_threadsafe(queue.put_nowait, output)

    def complete(self, request_id: str) -> None:
        """Thread-safe signal end of stream."""
        if request_id not in self._queues:
            logger.debug(
                f"[ResponseManager] complete failed: {request_id} not in queues"
            )
            return
        logger.debug(f"[ResponseManager] Completing {request_id}")
        queue, loop, _ = self._queues[request_id]
        loop.call_soon_threadsafe(queue.put_nowait, None)

    def shutdown(self) -> None:
        """Signal all active requests that the server is shutting down."""
        self._is_shutting_down = True
        # Create a copy of IDs to avoid dictionary mutation during iteration
        request_ids = list(self._queues.keys())
        logger.info(
            f"[ResponseManager] Shutting down {len(request_ids)} active streams"
        )
        for rid in request_ids:
            self.complete(rid)

    def fail(self, request_id: str, exception: Exception) -> None:
        """Thread-safe signal failure (not fully implemented for stream yet)."""
        # For now, just end the stream. Proper error propagation can be added later.
        self.complete(request_id)

    def has_request(self, request_id: str) -> bool:
        return request_id in self._queues
