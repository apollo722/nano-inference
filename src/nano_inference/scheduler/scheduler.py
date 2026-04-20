import math
import threading
import time
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
    def on_step_completed(self, request_ids: List[str]) -> None:
        """Release the 'running' status after an inference step."""
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

    @abstractmethod
    def get_stats(self) -> dict:
        """Get scheduler statistics."""
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
        self._last_batch_size = 0

        # Metrics tracking
        self._total_prompt_tokens = 0
        self._total_generation_tokens = 0
        self._start_time = time.time()

    def add_tasks(self, queries: List[GenerateQuery]) -> None:
        with self._lock:
            for query in queries:
                # Add to lookup if new
                if query.request_id not in self._queries:
                    self._queries[query.request_id] = query

                logger.debug(
                    f"[Scheduler] Adding task {query.request_id} to queue (stage={query.stage}, len={query.computed_length})"
                )

                # Routing logic:
                # 1. If explicitly in PREFILL stage OR RECOMPUTE stage AND hasn't processed anything yet (for prefill)
                #    OR needs recomputation, it's treated like a prefill.
                # 2. If it has already processed tokens (and NOT in RECOMPUTE), it MUST be a decode (rescheduled).
                # 3. Otherwise, follow the stage.
                if (
                    query.stage in (GenerationStage.PREFILL, GenerationStage.RECOMPUTE)
                    and query.computed_length == 0
                ):
                    if query not in self._prefill_waiting:
                        self._prefill_waiting.append(query)
                else:
                    if query not in self._decode_waiting:
                        self._decode_waiting.append(query)

    def schedule(self) -> List[GenerateQuery]:
        with self._lock:
            batch: List[GenerateQuery] = []

            # 1. Prioritize decode requests (latency)
            if self._decode_waiting:
                while self._decode_waiting and len(batch) < self.config.max_batch_size:
                    query = self._decode_waiting.popleft()

                    # Ensure KV block has space for at least 1 more token
                    if self.allocator:
                        if query.kv_cache_block is None:
                            logger.warning(
                                f"[Scheduler] req {query.request_id} has no KV block in decode stage. Skipping."
                            )
                            self._running.add(query.request_id)
                            self.finish_tasks([query.request_id])
                            continue

                        try:
                            # Attempt allocation
                            while query.kv_cache_block.free_slots < 1:
                                self.allocator.allocate_token(query.kv_cache_block)
                        except RuntimeError as e:
                            # KV OOM -> Preempt newest request (LIFO) and retry
                            logger.warning(
                                f"[Scheduler] KV OOM during decode for {query.request_id}: {e}. Triggering preemption."
                            )

                            # Find a preemptable request (can be current batch or already running)
                            # Simple Nano heuristic: preempt from _running or _decode_waiting
                            preempted = self._perform_lifo_preemption()
                            if preempted:
                                # Retry allocation for the same query
                                try:
                                    while query.kv_cache_block.free_slots < 1:
                                        self.allocator.allocate_token(
                                            query.kv_cache_block
                                        )
                                except RuntimeError:
                                    # Still failing? Defer this query.
                                    self._decode_waiting.appendleft(query)
                                    continue
                            else:
                                # No one left to preempt
                                self._decode_waiting.appendleft(query)
                                continue

                    self._running.add(query.request_id)
                    batch.append(query)

                # If we have decodes, return them exclusively (Homogeneous)
                if batch:
                    self._last_batch_size = len(batch)
                    return batch

            # 2. If no decodes, schedule prefill requests (throughput)
            prefill_added = 0
            while (
                self._prefill_waiting
                and len(batch) < self.config.max_batch_size
                and prefill_added < self.config.max_prefill_batch_size
            ):
                query = self._prefill_waiting.popleft()

                # Allocate initial blocks for prefill / recompute
                if self.allocator:
                    # In RECOMPUTE, we need enough space for Prompt + all previously generated tokens + 1 new token
                    # In PREFILL, just Prompt + 1 new token
                    sequence_len = len(query.generation_inputs.prompt_token_ids)
                    if query.stage == GenerationStage.RECOMPUTE:
                        sequence_len += len(query.output_token_ids)

                    try:
                        if query.kv_cache_block is None:
                            query.kv_cache_block = self.allocator.allocate(
                                sequence_len + 1
                            )
                        elif query.kv_cache_block.capacity < (sequence_len + 1):
                            needed = (sequence_len + 1) - query.kv_cache_block.capacity
                            num_blocks_needed = math.ceil(
                                needed / query.kv_cache_block.block_size
                            )
                            for _ in range(num_blocks_needed):
                                if not self.allocator.free_blocks:
                                    raise RuntimeError("Out of KV cache memory")
                                query.kv_cache_block.append_block(
                                    self.allocator.free_blocks.pop()
                                )
                    except RuntimeError as e:
                        logger.warning(
                            f"[Scheduler] Delaying prefill/recompute for {query.request_id}: {e}"
                        )
                        self._prefill_waiting.appendleft(query)
                        break

                self._running.add(query.request_id)
                batch.append(query)
                prefill_added += 1

            if batch:
                logger.debug(
                    f"[Scheduler] Scheduled batch of {len(batch)} (prefills={prefill_added})"
                )
            self._last_batch_size = len(batch)
            return batch

    def _perform_lifo_preemption(self) -> bool:
        """Find the newest request and preempt it to free up KV blocks.

        Nano strategy:
        1. Try to find a request in the decode waiting queue first (easiest).
        2. If empty, we could preempt from 'running', but in our current
           single-threaded step architecture, 'running' is only populated
           during the schedule() call for the NEXT step.
           So we actually look at self._queries and pick one that is not in the current batch.
        """
        # 1. Preempt from end of decode waiting (newest)
        if self._decode_waiting:
            q = self._decode_waiting.pop()
            self.preempt_request(q.request_id)
            return True

        # 2. Fallback: Preempt from prefill queue if somehow it has blocks (rare)
        if self._prefill_waiting:
            q = self._prefill_waiting.pop()
            if q.kv_cache_block:
                self.preempt_request(q.request_id)
                return True

        return False

    def preempt_request(self, request_id: str) -> None:
        """Preempt a request: free its KV cache and reset to RECOMPUTE stage."""
        query = self._queries.get(request_id)
        if not query:
            return

        logger.info(
            f"[Scheduler] Preempting request {request_id} (logical history preserved)"
        )

        # 1. Free KV Cache
        if self.allocator and query.kv_cache_block:
            self.allocator.free(query.kv_cache_block)
            query.kv_cache_block = None

        # 2. Set Query State for RECOMPUTE (do NOT clear output_token_ids)
        query.stage = GenerationStage.RECOMPUTE
        # We also reset computed_length to 0 as we'll rebuild from the prompt
        query.computed_length = 0

        # 3. Move back to prefill waiting queue (at the front so it's prioritized later)
        self._running.discard(request_id)
        if query not in self._prefill_waiting:
            self._prefill_waiting.appendleft(query)

    def on_step_completed(self, request_ids: List[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                self._running.discard(request_id)

    def finish_tasks(self, request_ids: List[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                logger.debug(f"[Scheduler] Cleanup request {request_id}")
                # 1. Remove from running set
                self._running.discard(request_id)

                # 2. Remove from waiting queues (if they were still there due to mixed batch logic or error)
                self._prefill_waiting = deque(
                    q for q in self._prefill_waiting if q.request_id != request_id
                )
                self._decode_waiting = deque(
                    q for q in self._decode_waiting if q.request_id != request_id
                )

                # 3. Free KV blocks and remove from lookup
                query = self._queries.pop(request_id, None)
                if query and self.allocator and query.kv_cache_block:
                    self.allocator.free(query.kv_cache_block)
                    query.kv_cache_block = None

            # Update last batch size after cleanup
            self._last_batch_size = len(self._running)

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
                self._last_batch_size = 0
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
            self._last_batch_size = len(self._running)

    def get_workload(self) -> int:
        with self._lock:
            return (
                len(self._prefill_waiting)
                + len(self._decode_waiting)
                + len(self._running)
            )

    def get_stats(self) -> dict:
        with self._lock:
            elapsed = time.time() - self._start_time
            avg_throughput = 0.0
            if elapsed > 0:
                avg_throughput = (
                    self._total_prompt_tokens + self._total_generation_tokens
                ) / elapsed

            stats = {
                "num_running": self._last_batch_size,
                "num_prefill_waiting": len(self._prefill_waiting),
                "num_decode_waiting": len(self._decode_waiting),
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_generation_tokens": self._total_generation_tokens,
                "avg_throughput_tps": round(avg_throughput, 2),
            }
            if self.allocator:
                stats["kv_utilization"] = self.allocator.utilization
            return stats

    def record_step(self, queries: List[GenerateQuery], num_new_tokens: int) -> None:
        """Record tokens processed in a step for throughput metrics."""
        with self._lock:
            for query in queries:
                if query.stage == GenerationStage.PREFILL:
                    # In prefill, the whole prompt is processed
                    self._total_prompt_tokens += len(
                        query.generation_inputs.prompt_token_ids
                    )
                    # Transition to DECODE for the next step
                    query.stage = GenerationStage.DECODE
                elif query.stage == GenerationStage.RECOMPUTE:
                    # In recompute, Prompt + existing output history is processed
                    # Note: output_token_ids already contains the token generated in this step
                    # so we subtract 1 to get the actual "recomputed" historical tokens.
                    self._total_prompt_tokens += len(
                        query.generation_inputs.prompt_token_ids
                    ) + (len(query.output_token_ids) - 1)
                    # Transition back to DECODE for the next step
                    query.stage = GenerationStage.DECODE

            # Generation tokens = one per query in the batch (since it's a step)
            # unless it's a mixed batch, but we simplify here.
            self._total_generation_tokens += num_new_tokens


class SimpleScheduler(OrcaScheduler):
    """Alias for OrcaScheduler to maintain compatibility while ensuring KV-awareness."""

    pass
