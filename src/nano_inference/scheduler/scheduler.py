import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from nano_inference.core.config import SchedulerConfig
from nano_inference.core.request import GenerateQuery, GenerationStage
from nano_inference.kv_cache import PagedKVCacheAllocator
from nano_inference.utils.logger import logger


@dataclass
class ScheduleTask:
    """Scheduling unit for the driver - wraps GenerateQuery with scheduling state."""

    generate_query: GenerateQuery
    is_finished: bool = False

    def __hash__(self) -> int:
        return hash(self.generate_query.request_id)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)


class SchedulerBase(ABC):
    """Abstract base for scheduling strategies (e.g., Orca).

    Input: individual GenerateQuery tasks.
    Output: batches of queries to run together.
    """

    def __init__(self):
        self._lock = threading.Lock()

    @abstractmethod
    def add_tasks(self, queries: List[GenerateQuery]) -> None:
        """Add new tasks to the scheduler."""
        ...

    @abstractmethod
    def schedule(self) -> List[GenerateQuery]:
        """Select next batch of queries to run."""
        ...

    @abstractmethod
    def finish_tasks(self, request_ids: List[str]) -> None:
        """Mark tasks as finished (remove from running set)."""
        ...

    @abstractmethod
    def abort_tasks(self, request_ids: Set[str]) -> None:
        """Abort tasks by request ID."""
        ...

    @abstractmethod
    def get_workload(self) -> int:
        """Get total number of pending/processing tasks."""
        ...


class OrcaScheduler(SchedulerBase):
    """Orca-style continuous batching scheduler.

    Has prefill/decode waiting queues and running queue.
    Phase 2: supports batching (default 32), Phase 3 will ad KV cache for speed.
    """

    def __init__(
        self, config: SchedulerConfig = None, allocator: PagedKVCacheAllocator = None
    ):
        super().__init__()
        self.config = config or SchedulerConfig()
        self.allocator = allocator
        self._prefill_waiting: deque[GenerateQuery] = deque()
        self._decode_waiting: deque[GenerateQuery] = deque()
        self._running: Set[str] = set()
        self._queries: Dict[str, GenerateQuery] = {}  # For lookup by ID

    def add_tasks(self, queries: List[GenerateQuery]) -> None:
        with self._lock:
            for query in queries:
                logger.debug(
                    f"[Scheduler] Adding task {query.request_id} (stage={query.stage})"
                )
                self._queries[query.request_id] = query
                if query.stage == GenerationStage.PREFILL:
                    self._prefill_waiting.append(query)
                else:
                    self._decode_waiting.append(query)

    def schedule(self) -> List[GenerateQuery]:
        with self._lock:
            batch: List[GenerateQuery] = []

            # 1. First, fill with decode requests (prioritize latency)
            while self._decode_waiting and len(batch) < self.config.max_batch_size:
                query = self._decode_waiting.popleft()

                # Ensure KV block has space for at least 1 more token
                if self.allocator:
                    try:
                        # Ensure we have enough blocks for history + 1 new token
                        safety_count = 0
                        while query.kv_cache_block.free_slots < 1:
                            self.allocator.allocate_token(query.kv_cache_block)
                            safety_count += 1
                            if safety_count > 100:
                                raise RuntimeError(
                                    f"Infinite loop in decode allocation for {query.request_id}"
                                )
                    except RuntimeError as e:
                        # Out of memory or infinite loop. Defer it.
                        logger.warning(
                            f"[Scheduler] Deferring decode for {query.request_id}: {e}"
                        )
                        self._decode_waiting.append(query)
                        continue

                self._running.add(query.request_id)
                batch.append(query)

            # 2. Then, if there is still room, add prefill requests (increase throughput)
            prefill_added = 0
            while (
                self._prefill_waiting
                and len(batch) < self.config.max_batch_size
                and prefill_added < self.config.max_prefill_batch_size
            ):
                query = self._prefill_waiting.popleft()

                # Allocate initial blocks for prefill
                if self.allocator:
                    prompt_len = len(query.generation_inputs.prompt_token_ids)
                    try:
                        # Allocate initial set (often just 1 block)
                        # Then loop to ensure we cover the entire prompt + 1 decode
                        if query.kv_cache_block is None:
                            query.kv_cache_block = self.allocator.allocate(1)

                        safety_count = 0
                        while query.kv_cache_block.capacity < (prompt_len + 1):
                            self.allocator.allocate_token(query.kv_cache_block)
                            safety_count += 1
                            if safety_count > 1000:
                                raise RuntimeError(
                                    f"Infinite loop in prefill allocation for {query.request_id}"
                                )

                    except RuntimeError as e:
                        # Out of memory or infinite loop. Put it back and stop prefilling.
                        logger.warning(
                            f"[Scheduler] Delaying prefill for {query.request_id}: {e}"
                        )
                        self._prefill_waiting.appendleft(query)
                        break

                self._running.add(query.request_id)
                batch.append(query)
                prefill_added += 1

            if batch:
                logger.debug(
                    f"[Scheduler] Scheduled batch of {len(batch)} "
                    f"(decodes={len(batch)-prefill_added}, prefills={prefill_added})"
                )
            return batch

    def finish_tasks(self, request_ids: List[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                if request_id in self._running:
                    logger.debug(f"[Scheduler] Finishing task {request_id}")
                    self._running.discard(request_id)

                    # Free KV blocks
                    query = self._queries.pop(request_id, None)
                    if query and self.allocator:
                        self.allocator.free(query.kv_cache_block)

    def abort_tasks(self, request_ids: Set[str]) -> None:
        with self._lock:
            if not request_ids:
                logger.debug("[Scheduler] Aborting ALL tasks")
                # Free all known queries
                if self.allocator:
                    for q in self._queries.values():
                        self.allocator.free(q.kv_cache_block)

                self._prefill_waiting.clear()
                self._decode_waiting.clear()
                self._running.clear()
                self._queries.clear()
                return

            logger.debug(f"[Scheduler] Aborting tasks: {request_ids}")
            self._prefill_waiting = deque(
                q for q in self._prefill_waiting if q.request_id not in request_ids
            )
            self._decode_waiting = deque(
                q for q in self._decode_waiting if q.request_id not in request_ids
            )
            for request_id in request_ids:
                self._running.discard(request_id)
                query = self._queries.pop(request_id, None)
                if query and self.allocator:
                    self.allocator.free(query.kv_cache_block)

    def get_workload(self) -> int:
        with self._lock:
            return (
                len(self._prefill_waiting)
                + len(self._decode_waiting)
                + len(self._running)
            )


class SimpleScheduler(OrcaScheduler):
    """Alias for OrcaScheduler to maintain compatibility while ensuring KV-awareness."""

    pass
