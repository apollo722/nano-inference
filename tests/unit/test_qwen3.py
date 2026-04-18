import torch
from nano_inference.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig


def test_qwen3_model_forward_pass():
    config = Qwen3ModelConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_layers=2,
        num_kv_heads=2,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        logits = model(input_ids)

    # [batch, seq, vocab]
    assert logits.shape == (1, 10, 100)


def test_qwen3_model_tie_word_embeddings():
    config = Qwen3ModelConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_layers=1,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config)

    input_ids = torch.randint(0, 100, (1, 2))
    # Should not crash and use shared weights
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, 2, 100)
    assert not hasattr(model.logits_processor, "lm_head")
