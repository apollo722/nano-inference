import pytest
import torch
from fastapi.testclient import TestClient
from nano_inference.api.server import create_app
from nano_inference.core.config import ModelConfig, RuntimeConfig, SchedulerConfig
from nano_inference.kv_cache.allocator import PagedKVCacheAllocator
from nano_inference.scheduler import OrcaScheduler


def test_scheduler_get_stats():
    # Setup allocator and scheduler
    allocator = PagedKVCacheAllocator(
        num_blocks=10, block_size=4, num_heads=1, head_dim=1, num_layers=1, device="cpu"
    )
    scheduler = OrcaScheduler(allocator=allocator)

    # Check initial stats
    stats = scheduler.get_stats()
    assert stats["num_running"] == 0
    assert stats["kv_utilization"] == 0.0

    # Simulate scheduling and step recording
    block = allocator.allocate(num_tokens=4)  # Uses 1 block
    scheduler._last_batch_size = 1
    scheduler._running.add("test-req")
    # Simulate a prefill step with 4 tokens
    from nano_inference.core.request import (
        GenerateQuery,
        GenerationInputs,
        SamplingParams,
    )

    q = GenerateQuery(
        request_id="test-req",
        generation_inputs=GenerationInputs(prompt_token_ids=[1, 2, 3, 4]),
        sampling_params=SamplingParams(),
        eos_token_id=0,
        arrival_time=0.0,
    )
    scheduler.record_step([q], 1)

    stats = scheduler.get_stats()
    assert stats["num_running"] == 1
    assert stats["kv_utilization"] == pytest.approx(0.1)  # 1/10 blocks
    assert stats["total_prompt_tokens"] == 4
    assert stats["total_generation_tokens"] == 1
    assert stats["avg_throughput_tps"] > 0


def test_metrics_endpoint():
    # Mock config
    config = RuntimeConfig(
        model=ModelConfig(model_dir="test-models/Qwen--Qwen3-0.6B", device="cpu"),
        scheduler=SchedulerConfig(),
    )

    app = create_app(config)
    client = TestClient(app)

    # Manually set state for the test to avoid lifespan model loading
    from unittest.mock import MagicMock

    mock_state = MagicMock()
    mock_state.scheduler.get_stats.return_value = {"num_running": 0}
    app.state.state = mock_state

    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["num_running"] == 0
