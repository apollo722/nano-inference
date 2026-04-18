from dataclasses import dataclass

import pytest
import torch
from nano_inference.core.config import ModelConfig
from nano_inference.model_loader.qwen3 import Qwen3Loader


@dataclass
class MockHFConfig:
    vocab_size: int = 1000
    hidden_size: int = 128
    intermediate_size: int = 256
    num_attention_heads: int = 8
    num_hidden_layers: int = 2
    num_key_value_heads: int = 2
    model_type: str = "qwen3"


def test_qwen3_loader_config_conversion():
    loader = Qwen3Loader()
    hf_config = MockHFConfig()
    runtime_config = ModelConfig(model_dir="dummy", max_length=2048)

    decoder_config = loader.to_decoder_config(hf_config, runtime_config)

    assert decoder_config.vocab_size == 1000
    assert decoder_config.hidden_size == 128
    assert decoder_config.num_heads == 8
    assert decoder_config.max_length == 2048


def test_qwen3_loader_remap_state_dict():
    loader = Qwen3Loader()
    hf_config = MockHFConfig()
    decoder_config = loader.to_decoder_config(hf_config)

    # Mock a state dict with one key
    hf_state_dict = {
        "model.embed_tokens.weight": torch.randn(1000, 128),
        "model.layers.0.input_layernorm.weight": torch.randn(128),
        "lm_head.weight": torch.randn(1000, 128),
    }

    remapped = loader.remap_state_dict(hf_state_dict, decoder_config)

    assert "model.embed_tokens.weight" in remapped
    assert "model.layers.0.input_norm.weight" in remapped
    assert "logits_processor.lm_head.weight" in remapped
    # Verify the values are the same
    torch.testing.assert_close(
        remapped["model.embed_tokens.weight"],
        hf_state_dict["model.embed_tokens.weight"],
    )


def test_qwen3_loader_remap_state_dict_filters_unknown_keys():
    loader = Qwen3Loader()
    hf_config = MockHFConfig()
    decoder_config = loader.to_decoder_config(hf_config)

    hf_state_dict = {
        "model.embed_tokens.weight": torch.randn(1000, 128),
        "unknown_key": torch.randn(10),  # Should be ignored
    }

    remapped = loader.remap_state_dict(hf_state_dict, decoder_config)
    assert "model.embed_tokens.weight" in remapped
    assert "unknown_key" not in remapped
    assert len(remapped) == 1
