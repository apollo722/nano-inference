from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from nano_inference.core.config import ModelConfig
from nano_inference.model_loader.base import BaseModelLoader
from nano_inference.models.qwen3 import Qwen3ForCausalLM, Qwen3ModelConfig


class Qwen3Loader(BaseModelLoader):
    @staticmethod
    def _config_to_dict(hf_config) -> dict:
        if hasattr(hf_config, "to_dict"):
            return hf_config.to_dict()
        return {
            key: getattr(hf_config, key)
            for key in dir(hf_config)
            if not key.startswith("_") and not callable(getattr(hf_config, key))
        }

    def can_load(self, hf_config) -> bool:
        cfg = self._config_to_dict(hf_config)
        model_type = (cfg.get("model_type") or "").lower()
        if model_type == "qwen3":
            return True
        for arch in cfg.get("architectures") or []:
            if "qwen3" in (arch or "").lower():
                return True
        return False

    def to_decoder_config(
        self,
        hf_config,
        runtime_config: ModelConfig | None = None,
    ) -> Qwen3ModelConfig:
        cfg = self._config_to_dict(hf_config)
        kwargs = {
            "vocab_size": cfg["vocab_size"],
            "hidden_size": cfg["hidden_size"],
            "intermediate_size": cfg["intermediate_size"],
            "num_heads": cfg["num_attention_heads"],
            "num_layers": cfg["num_hidden_layers"],
            "num_kv_heads": cfg.get("num_key_value_heads"),
            "head_dim": cfg.get("head_dim"),
            "use_qk_norm": True,  # qwen3 always uses qk norm
            "max_length": cfg.get("max_position_embeddings", 4096),
            "rms_norm_eps": cfg.get("rms_norm_eps", 1e-6),
            "rope_base": cfg.get("rope_theta", 1000000.0),  # qwen3 uses 1e6
            "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
        }

        if runtime_config is not None:
            kwargs["max_length"] = runtime_config.max_length

        return Qwen3ModelConfig(**kwargs)

    @staticmethod
    def iter_hf_to_decoder_key_mapping(num_layers: int) -> Iterable[Tuple[str, str]]:
        yield "model.embed_tokens.weight", "model.embed_tokens.weight"
        yield "model.norm.weight", "model.norm.weight"
        yield "lm_head.weight", "logits_processor.lm_head.weight"

        for layer_idx in range(num_layers):
            hf_prefix = f"model.layers.{layer_idx}"
            our_prefix = f"model.layers.{layer_idx}"
            yield (
                f"{hf_prefix}.input_layernorm.weight",
                f"{our_prefix}.input_norm.weight",
            )
            yield (
                f"{hf_prefix}.post_attention_layernorm.weight",
                f"{our_prefix}.post_attention_norm.weight",
            )
            yield (
                f"{hf_prefix}.self_attn.q_proj.weight",
                f"{our_prefix}.self_attn.q_proj.weight",
            )
            yield (
                f"{hf_prefix}.self_attn.k_proj.weight",
                f"{our_prefix}.self_attn.k_proj.weight",
            )
            yield (
                f"{hf_prefix}.self_attn.v_proj.weight",
                f"{our_prefix}.self_attn.v_proj.weight",
            )
            yield (
                f"{hf_prefix}.self_attn.o_proj.weight",
                f"{our_prefix}.self_attn.o_proj.weight",
            )
            yield (
                f"{hf_prefix}.mlp.gate_proj.weight",
                f"{our_prefix}.mlp.gate_proj.weight",
            )
            yield f"{hf_prefix}.mlp.up_proj.weight", f"{our_prefix}.mlp.up_proj.weight"
            yield (
                f"{hf_prefix}.mlp.down_proj.weight",
                f"{our_prefix}.mlp.down_proj.weight",
            )

    def remap_state_dict(
        self,
        hf_state_dict: Dict[str, torch.Tensor],
        decoder_config: Qwen3ModelConfig,
    ) -> Dict[str, torch.Tensor]:
        remapped: Dict[str, torch.Tensor] = {}
        for hf_key, our_key in self.iter_hf_to_decoder_key_mapping(
            decoder_config.num_layers
        ):
            # Skip lm_head.weight if tie_word_embeddings=True (we'll tie it anyway)
            if (
                our_key == "logits_processor.lm_head.weight"
                and decoder_config.tie_word_embeddings
            ):
                continue

            if hf_key in hf_state_dict:
                remapped[our_key] = hf_state_dict[hf_key]

        # Load q_norm and k_norm weights for qwen3
        for layer_idx in range(decoder_config.num_layers):
            hf_prefix = f"model.layers.{layer_idx}.self_attn"
            our_prefix = f"model.layers.{layer_idx}.self_attn"
            q_norm_key = f"{hf_prefix}.q_norm.weight"
            k_norm_key = f"{hf_prefix}.k_norm.weight"
            if q_norm_key in hf_state_dict:
                remapped[f"{our_prefix}.q_norm.weight"] = hf_state_dict[q_norm_key]
            if k_norm_key in hf_state_dict:
                remapped[f"{our_prefix}.k_norm.weight"] = hf_state_dict[k_norm_key]

        return remapped

    def build_model(self, decoder_config: Qwen3ModelConfig) -> nn.Module:
        return Qwen3ForCausalLM(decoder_config)
