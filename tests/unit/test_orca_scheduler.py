import time

from nano_inference.core.request import (
    GenerateQuery,
    GenerationInputs,
    GenerationStage,
    Request,
    SamplingParams,
)
from nano_inference.scheduler import OrcaScheduler


def test_orca_scheduler_prioritizes_decode_over_prefill():
    scheduler = OrcaScheduler(max_batch_size=2)

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
