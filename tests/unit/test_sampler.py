import torch
import pytest
from nano_inference.sampling.sampler import Sampler
from nano_inference.core.sampling import SamplingParams

@pytest.fixture
def sampler():
    return Sampler()

def test_sampler_greedy_selection(sampler):
    logits = torch.tensor([-1.0, 5.0, 2.0, -0.5])
    params = SamplingParams(temperature=0.0)
    # index 1 is max (5.0)
    selected = sampler.select(logits, generated_ids=[], sampling_params=params)
    assert selected == 1

def test_sampler_repetition_penalty(sampler):
    # Logits: high score for index 1
    logits = torch.tensor([0.0, 10.0, 0.0])
    # Case 1: No penalty (1.0)
    params = SamplingParams(temperature=0.0, repetition_penalty=1.0)
    assert sampler.select(logits, generated_ids=[1], sampling_params=params) == 1
    
    # Case 2: High penalty (e.g., 20.0)
    # Index 1 is positive (10.0), so it will be 10.0 / 20.0 = 0.5
    # Index 0 and 2 are 0.0, so they will stay 0.0. 0.5 is still max among [0.0, 0.5, 0.0]
    # Let's use a penalty that makes it smaller than others
    params = SamplingParams(temperature=0.0, repetition_penalty=100.0)
    # 10.0 / 100.0 = 0.1. Wait, 0.1 is still bigger than 0.0. 
    # Let's make index 0 bigger than 0.1
    logits = torch.tensor([1.0, 10.0, 1.0])
    # Index 1 was already generated, so 10.0 / 100.0 = 0.1. Now index 0 or 2 (1.0) should be max.
    selected = sampler.select(logits, generated_ids=[1], sampling_params=params)
    assert selected in [0, 2]

def test_sampler_top_k(sampler):
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # Top-K = 2 -> only consider 4.0 and 5.0. 
    # With temp=1.0, we can't be sure, but let's test the filtering logic directly if possible.
    # We can check if it never picks 1.0, 2.0, or 3.0.
    params = SamplingParams(temperature=1.0, top_k=2)
    # Mocking multinomial to verify behavior is hard, but we can run it many times.
    selections = {sampler.select(logits, generated_ids=[], sampling_params=params) for _ in range(20)}
    assert selections.issubset({3, 4})
    assert 0 not in selections
    assert 1 not in selections
    assert 2 not in selections
