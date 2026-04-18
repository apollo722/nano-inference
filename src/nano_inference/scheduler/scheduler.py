import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import List, Set, Tuple

from nano_inference.core.request import GenerateQuery, GenerationStage


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


class SimpleScheduler(SchedulerBase):
    """Simple FIFO scheduler - batches all available tasks (one per step for Phase 2)."""

    def __init__(self):
        super().__init__()
        self._waiting: deque[GenerateQuery] = deque()
        self._running: Set[str] = set()

    def add_tasks(self, queries: List[GenerateQuery]) -> None:
        with self._lock:
            for query in queries:
                self._waiting.append(query)

    def schedule(self) -> List[GenerateQuery]:
        with self._lock:
            if not self._waiting:
                return []
            # Phase 2: one query at a time (we'll upgrade to real batching later)
            query = self._waiting.popleft()
            self._running.add(query.request_id)
            return [query]

    def finish_tasks(self, request_ids: List[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                self._running.discard(request_id)

    def abort_tasks(self, request_ids: Set[str]) -> None:
        with self._lock:
            self._waiting = deque(
                q for q in self._waiting if q.request_id not in request_ids
            )
            for request_id in request_ids:
                self._running.discard(request_id)

    def get_workload(self) -> int:
        with self._lock:
            return len(self._waiting) + len(self._running)


class OrcaScheduler(SchedulerBase):
    """Orca-style continuous batching scheduler.

    Has prefill/decode waiting queues and running queue.
    Phase 2: supports batching (default 32), Phase 3 will ad KV cache for speed.
    """

    def __init__(self, max_batch_size: int = 32):
        super().__init__()
        self.max_batch_size = max_batch_size
        self._prefill_waiting: deque[GenerateQuery] = deque()
        self._decode_waiting: deque[GenerateQuery] = deque()
        self._running: Set[str] = set()

    def add_tasks(self, queries: List[GenerateQuery]) -> None:
        with self._lock:
            for query in queries:
                if query.stage == GenerationStage.PREFILL:
                    self._prefill_waiting.append(query)
                else:
                    self._decode_waiting.append(query)

    def schedule(self) -> List[GenerateQuery]:
        with self._lock:
            batch: List[GenerateQuery] = []

            while self._decode_waiting and len(batch) < self.max_batch_size:
                query = self._decode_waiting.popleft()
                self._running.add(query.request_id)
                batch.append(query)
            if batch:
                return batch

            while self._prefill_waiting and len(batch) < self.max_batch_size:
                query = self._prefill_waiting.popleft()
                self._running.add(query.request_id)
                batch.append(query)
            return batch

    def finish_tasks(self, request_ids: List[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                self._running.discard(request_id)

    def abort_tasks(self, request_ids: Set[str]) -> None:
        with self._lock:
            self._prefill_waiting = deque(
                q for q in self._prefill_waiting if q.request_id not in request_ids
            )
            self._decode_waiting = deque(
                q for q in self._decode_waiting if q.request_id not in request_ids
            )
            for request_id in request_ids:
                self._running.discard(request_id)

    def get_workload(self) -> int:
        with self._lock:
            return (
                len(self._prefill_waiting)
                + len(self._decode_waiting)
                + len(self._running)
            )
