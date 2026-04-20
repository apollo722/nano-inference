"""Model loader for Qwen2.5-VL (vision-language model)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from nano_inference.core.config import ModelConfig
from nano_inference.model_loader.base import BaseModelLoader
from nano_inference.model_loader.qwen3 import Qwen3Loader
from nano_inference.models.qwen25_vl import (
    Qwen25VLForConditionalGeneration,
    Qwen25VLModelConfig,
)


class Qwen25VLLoader(BaseModelLoader):
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
        if "qwen2_5_vl" in model_type or "qwen2.5_vl" in model_type:
            return True
        for arch in cfg.get("architectures") or []:
            if (
                "qwen2_5vl" in (arch or "").lower()
                or "qwen2.5vl" in (arch or "").lower()
            ):
                return True
        return False

    def to_decoder_config(
        self,
        hf_config,
        runtime_config: ModelConfig | None = None,
    ) -> Qwen25VLModelConfig:
        cfg = self._config_to_dict(hf_config)

        # Text config may be nested under text_config for Qwen2.5-VL
        text_cfg = cfg.get("text_config") or cfg

        # mrope_section lives under rope_scaling / rope_parameters depending on HF version
        rope_params = (
            text_cfg.get("rope_scaling") or text_cfg.get("rope_parameters") or {}
        )
        if isinstance(rope_params, dict):
            mrope_section_raw = rope_params.get("mrope_section", [16, 24, 24])
            rope_theta = rope_params.get(
                "rope_theta", text_cfg.get("rope_theta", 1000000.0)
            )
        else:
            mrope_section_raw = [16, 24, 24]
            rope_theta = text_cfg.get("rope_theta", 1000000.0)

        mrope_section = tuple(int(x) for x in mrope_section_raw)

        # Vision config
        vision_cfg = cfg.get("vision_config") or {}
        if hasattr(hf_config, "vision_config"):
            vc = hf_config.vision_config
            vision_cfg = (
                self._config_to_dict(vc)
                if hasattr(vc, "to_dict") or hasattr(vc, "__dict__")
                else vision_cfg
            )

        kwargs = dict(
            # text decoder
            vocab_size=text_cfg.get("vocab_size", cfg.get("vocab_size")),
            hidden_size=text_cfg.get("hidden_size", cfg.get("hidden_size")),
            intermediate_size=text_cfg.get(
                "intermediate_size", cfg.get("intermediate_size")
            ),
            num_heads=text_cfg.get(
                "num_attention_heads", cfg.get("num_attention_heads")
            ),
            num_layers=text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers")),
            num_kv_heads=text_cfg.get(
                "num_key_value_heads", cfg.get("num_key_value_heads")
            ),
            head_dim=text_cfg.get("head_dim", cfg.get("head_dim")),
            use_qk_norm=False,  # Qwen2.5-VL does not use QK-norm
            max_length=text_cfg.get(
                "max_position_embeddings", cfg.get("max_position_embeddings", 4096)
            ),
            rms_norm_eps=text_cfg.get("rms_norm_eps", cfg.get("rms_norm_eps", 1e-6)),
            rope_base=rope_theta,
            mrope_section=mrope_section,
            tie_word_embeddings=text_cfg.get(
                "tie_word_embeddings", cfg.get("tie_word_embeddings", False)
            ),
            attention_bias=bool(
                text_cfg.get("attention_bias", cfg.get("attention_bias", True))
            ),
            # vision encoder
            vision_hidden_size=vision_cfg.get("hidden_size", 1280),
            vision_num_heads=vision_cfg.get("num_heads", 16),
            vision_depth=vision_cfg.get("depth", 32),
            vision_intermediate_size=vision_cfg.get("intermediate_size", 3420),
            vision_patch_size=vision_cfg.get("patch_size", 14),
            vision_temporal_patch_size=vision_cfg.get("temporal_patch_size", 2),
            vision_spatial_merge_size=vision_cfg.get("spatial_merge_size", 2),
            vision_window_size=vision_cfg.get("window_size", 112),
            vision_fullatt_block_indexes=tuple(
                vision_cfg.get("fullatt_block_indexes", [7, 15, 23, 31])
            ),
            image_token_id=int(cfg.get("image_token_id", 151655)),
        )

        if runtime_config is not None:
            kwargs["max_length"] = runtime_config.max_length

        return Qwen25VLModelConfig(**kwargs)

    @staticmethod
    def iter_hf_to_decoder_key_mapping(
        num_layers: int,
        vision_depth: int = 32,
        attention_bias: bool = True,
    ) -> Iterable[Tuple[str, str]]:
        # ── Text decoder (HF wraps LLM under "language_model.*") ────────────────
        yield "language_model.embed_tokens.weight", "model.embed_tokens.weight"
        yield "language_model.norm.weight", "model.norm.weight"
        yield "language_model.lm_head.weight", "logits_processor.lm_head.weight"

        for i in range(num_layers):
            hp = f"language_model.layers.{i}"
            op = f"model.layers.{i}"
            yield f"{hp}.input_layernorm.weight", f"{op}.input_norm.weight"
            yield (
                f"{hp}.post_attention_layernorm.weight",
                f"{op}.post_attention_norm.weight",
            )
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                yield f"{hp}.self_attn.{proj}.weight", f"{op}.self_attn.{proj}.weight"
            if attention_bias:
                for proj in ("q_proj", "k_proj", "v_proj"):
                    yield f"{hp}.self_attn.{proj}.bias", f"{op}.self_attn.{proj}.bias"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield f"{hp}.mlp.{proj}.weight", f"{op}.mlp.{proj}.weight"

        # ── Vision encoder (HF "visual.*" → our "model.visual.*") ─────────────
        yield "visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.weight"
        # patch_embed.proj has no bias in Qwen2.5-VL

        for i in range(vision_depth):
            vp = f"visual.blocks.{i}"
            op = f"model.visual.blocks.{i}"
            yield f"{vp}.norm1.weight", f"{op}.norm1.weight"
            yield f"{vp}.norm2.weight", f"{op}.norm2.weight"
            yield f"{vp}.attn.qkv.weight", f"{op}.attn.qkv.weight"
            yield f"{vp}.attn.qkv.bias", f"{op}.attn.qkv.bias"
            yield f"{vp}.attn.proj.weight", f"{op}.attn.proj.weight"
            yield f"{vp}.attn.proj.bias", f"{op}.attn.proj.bias"
            yield f"{vp}.mlp.gate_proj.weight", f"{op}.mlp.gate_proj.weight"
            yield f"{vp}.mlp.gate_proj.bias", f"{op}.mlp.gate_proj.bias"
            yield f"{vp}.mlp.up_proj.weight", f"{op}.mlp.up_proj.weight"
            yield f"{vp}.mlp.up_proj.bias", f"{op}.mlp.up_proj.bias"
            yield f"{vp}.mlp.down_proj.weight", f"{op}.mlp.down_proj.weight"
            yield f"{vp}.mlp.down_proj.bias", f"{op}.mlp.down_proj.bias"

        yield "visual.merger.ln_q.weight", "model.visual.merger.ln_q.weight"
        yield "visual.merger.mlp.0.weight", "model.visual.merger.mlp.0.weight"
        yield "visual.merger.mlp.0.bias", "model.visual.merger.mlp.0.bias"
        yield "visual.merger.mlp.2.weight", "model.visual.merger.mlp.2.weight"
        yield "visual.merger.mlp.2.bias", "model.visual.merger.mlp.2.bias"

    def remap_state_dict(
        self,
        hf_state_dict: Dict[str, torch.Tensor],
        decoder_config: Qwen25VLModelConfig,
    ) -> Dict[str, torch.Tensor]:
        remapped: Dict[str, torch.Tensor] = {}
        for hf_key, our_key in self.iter_hf_to_decoder_key_mapping(
            num_layers=decoder_config.num_layers,
            vision_depth=decoder_config.vision_depth,
            attention_bias=decoder_config.attention_bias,
        ):
            if (
                our_key == "logits_processor.lm_head.weight"
                and decoder_config.tie_word_embeddings
            ):
                continue
            if hf_key in hf_state_dict:
                remapped[our_key] = hf_state_dict[hf_key]

        # HF patch_embed uses Conv3d weight (out, C, T, H, W); we use Conv2d (out, C*T, H, W)
        pe_key = "model.visual.patch_embed.proj.weight"
        if pe_key in remapped and remapped[pe_key].dim() == 5:
            w = remapped[pe_key]
            remapped[pe_key] = w.reshape(
                w.shape[0], w.shape[1] * w.shape[2], w.shape[3], w.shape[4]
            )

        return remapped

    def build_model(self, decoder_config: Qwen25VLModelConfig) -> nn.Module:
        return Qwen25VLForConditionalGeneration(decoder_config)
