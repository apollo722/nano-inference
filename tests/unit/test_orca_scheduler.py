import time

from nano_inference.core.config import SchedulerConfig
from nano_inference.core.request import (
    GenerateQuery,
    GenerationInputs,
    GenerationStage,
    Request,
    SamplingParams,
)
from nano_inference.scheduler import OrcaScheduler


def test_orca_scheduler_prioritizes_decode_over_prefill():
    config = SchedulerConfig(max_batch_size=2, max_prefill_batch_size=2)
    scheduler = OrcaScheduler(config=config)

    # Add 3 prefill queries
    prefill_queries = []
    for i in range(3):
        req = Request(
            request_id=f"prefill-{i}",
            generation_inputs=GenerationInputs(prompt_token_ids=[1, 2, 3]),
            sampling_params=SamplingParams(),
            eos_token_id=0,
            arrival_time=time.time(),
        )
        prefill_queries.append(GenerateQuery.from_request(req))

    # Add 2 decode queries
    decode_queries = []
    for i in range(2):
        req = Request(
            request_id=f"decode-{i}",
            generation_inputs=GenerationInputs(prompt_token_ids=[1, 2, 3]),
            sampling_params=SamplingParams(),
            eos_token_id=0,
            arrival_time=time.time(),
        )
        query = GenerateQuery.from_request(req)
        query.stage = GenerationStage.DECODE  # Manually set stage
        decode_queries.append(query)

    scheduler.add_tasks(prefill_queries)
    scheduler.add_tasks(decode_queries)

    # First schedule should pick decode queries first
    batch1 = scheduler.schedule()
    assert len(batch1) == 2
    assert all(q.request_id.startswith("decode") for q in batch1)

    # Mark decode queries as finished
    scheduler.finish_tasks([q.request_id for q in batch1])

    # Next schedule should pick prefill queries
    batch2 = scheduler.schedule()
    assert len(batch2) == 2
    assert all(q.request_id.startswith("prefill") for q in batch2)


def test_orca_scheduler_respects_max_prefill_limit():
    # max_prefill_batch_size=1, max_batch_size=4
    config = SchedulerConfig(max_batch_size=4, max_prefill_batch_size=1)
    scheduler = OrcaScheduler(config=config)

    # 1. Add 2 prefill queries
    q1 = GenerateQuery.from_request(
        Request("q1", GenerationInputs([1]), SamplingParams(), 0, 0)
    )
    q2 = GenerateQuery.from_request(
        Request("q2", GenerationInputs([1]), SamplingParams(), 0, 0)
    )
    scheduler.add_tasks([q1, q2])

    # 2. Schedule should only pick 1 prefill
    batch = scheduler.schedule()
    assert len(batch) == 1
    assert batch[0].request_id == "q1"
    scheduler.finish_tasks(["q1"])

    # 3. Add a decode task
    q3 = GenerateQuery.from_request(
        Request("q3", GenerationInputs([1]), SamplingParams(), 0, 0)
    )
    q3.stage = GenerationStage.DECODE
    scheduler.add_tasks([q3])

    # 4. Schedule should pick only the decode task (Homogeneous)
    batch2 = scheduler.schedule()
    assert len(batch2) == 1
    assert batch2[0].request_id == "q3"

    # 5. Next schedule should pick the remaining prefill
    scheduler.finish_tasks(["q3"])
    batch3 = scheduler.schedule()
    assert len(batch3) == 1
    assert batch3[0].request_id == "q2"
