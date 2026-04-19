from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from nano_inference.core.config import ModelConfig
from nano_inference.core.context import AttentionMetadata
from nano_inference.layers import (
    NaiveLogitsProcessor,
    NaiveRMSNorm,
    NaiveSwiGLUMLP,
    NaiveTokenEmbedding,
    PagedCausalSelfAttention,
)
from nano_inference.utils.pickle_ops import dump_output_pickle


@dataclass
class Qwen3ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_layers: int
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    use_qk_norm: bool = False
    max_length: int = 4096
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    tie_word_embeddings: bool = False

    @classmethod
    def from_runtime_config(
        cls, runtime_config: ModelConfig, **kwargs
    ) -> "Qwen3ModelConfig":
        # Reuse shared runtime limits from core ModelConfig while keeping
        # decoder architecture fields separate from device/loading concerns.
        return cls(max_length=runtime_config.max_length, **kwargs)


class Qwen3DecoderBlock(nn.Module):
    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        self.input_norm = NaiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PagedCausalSelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            use_qk_norm=config.use_qk_norm,
            rope_base=config.rope_base,
        )
        self.post_attention_norm = NaiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = NaiveSwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    @dump_output_pickle(name="qwen3-decoder-block")
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        attn_out = self.self_attn(
            self.input_norm(x),
            position_ids=position_ids,
            attention_mask=attention_mask,
            metadata=metadata,
        )
        x = x + attn_out

        mlp_out = self.mlp(self.post_attention_norm(x))
        x = x + mlp_out
        return x


class Qwen3TransformerModel(nn.Module):
    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = NaiveTokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderBlock(config) for _ in range(config.num_layers)]
        )
        # Set layer indices for Paged Attention
        for i, layer in enumerate(self.layers):
            layer.self_attn.layer_idx = i
        self.norm = NaiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(
                x,
                position_ids=position_ids,
                attention_mask=attention_mask,
                metadata=metadata,
            )

        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3TransformerModel(config)
        self.logits_processor = NaiveLogitsProcessor(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            tie_word_embeddings=config.tie_word_embeddings,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            metadata=metadata,
        )

        return self.logits_processor(
            hidden_states,
            embed_tokens_weight=self.model.embed_tokens.weight
            if self.config.tie_word_embeddings
            else None,
        )
