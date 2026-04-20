"""Qwen2.5-VL vision-language model.

Decoder is structurally identical to Qwen3 except:
- Decoder attention uses Qwen25VLDecoderAttention (mRoPE instead of 1D RoPE).
- TransformerModel holds a Qwen25VisionTransformer vision encoder.
- Prefill forward stitches ViT output into text embeddings at image_token_id positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from nano_inference.core.context import AttentionMetadata
from nano_inference.layers import (
    NaiveLogitsProcessor,
    NaiveRMSNorm,
    NaiveSwiGLUMLP,
    NaiveTokenEmbedding,
    Qwen25VisionTransformer,
    Qwen25VLDecoderAttention,
)


@dataclass
class Qwen25VLModelConfig:
    # ── Text decoder ────────────────────────────────────────────────────────
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
    rope_base: float = 1000000.0
    mrope_section: Tuple[int, int, int] = field(default_factory=lambda: (16, 24, 24))
    tie_word_embeddings: bool = False
    attention_bias: bool = True  # Qwen2.5-VL text decoder has q/k/v biases

    # ── Vision encoder ───────────────────────────────────────────────────────
    image_token_id: int = 151655  # <|image_pad|>; populated from HF config by loader
    vision_hidden_size: int = 1280
    vision_num_heads: int = 16
    vision_depth: int = 32
    vision_intermediate_size: int = 3420
    vision_patch_size: int = 14
    vision_temporal_patch_size: int = 2
    vision_spatial_merge_size: int = 2
    vision_window_size: int = 112
    vision_fullatt_block_indexes: Tuple[int, ...] = field(
        default_factory=lambda: (7, 15, 23, 31)
    )


class Qwen25VLDecoderBlock(nn.Module):
    def __init__(self, config: Qwen25VLModelConfig):
        super().__init__()
        self.input_norm = NaiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen25VLDecoderAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            use_qk_norm=config.use_qk_norm,
            qkv_bias=config.attention_bias,
            rope_base=config.rope_base,
            mrope_section=config.mrope_section,
        )
        self.post_attention_norm = NaiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = NaiveSwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

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
        x = x + self.mlp(self.post_attention_norm(x))
        return x


class Qwen25VLTransformerModel(nn.Module):
    def __init__(self, config: Qwen25VLModelConfig):
        super().__init__()
        self.config = config

        self.visual = Qwen25VisionTransformer(
            hidden_size=config.vision_hidden_size,
            num_heads=config.vision_num_heads,
            depth=config.vision_depth,
            intermediate_size=config.vision_intermediate_size,
            out_hidden_size=config.hidden_size,
            patch_size=config.vision_patch_size,
            temporal_patch_size=config.vision_temporal_patch_size,
            spatial_merge_size=config.vision_spatial_merge_size,
            window_size=config.vision_window_size,
            fullatt_block_indexes=config.vision_fullatt_block_indexes,
        )

        self.embed_tokens = NaiveTokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [Qwen25VLDecoderBlock(config) for _ in range(config.num_layers)]
        )
        for i, layer in enumerate(self.layers):
            layer.self_attn.layer_idx = i
        self.norm = NaiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Set by loader after construction (holds HF image_processor)
        self.image_processor: Optional[Any] = None

    def _encode_images(
        self,
        images: List[Any],
        image_grid_thw: List[Tuple[int, int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Preprocess images and run through the vision encoder."""
        assert (
            self.image_processor is not None
        ), "image_processor must be set on the model before VLM inference"
        result = self.image_processor(
            images=images,
            return_tensors="pt",
        )
        pixel_values = result["pixel_values"].to(device=device, dtype=dtype)
        # HF image_processor returns (total_patches, C*T*H*W) — reshape to 4D for Conv2d
        if pixel_values.dim() == 2:
            ps = self.config.vision_patch_size
            tps = self.config.vision_temporal_patch_size
            pixel_values = pixel_values.reshape(-1, 3 * tps, ps, ps)
        return self.visual(pixel_values, image_grid_thw)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[AttentionMetadata] = None,
        images: Optional[List[Any]] = None,
        image_grid_thw: Optional[List[Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden = self.embed_tokens(input_ids)

        if images and image_grid_thw:
            vision_out = self._encode_images(
                images, image_grid_thw, device=hidden.device, dtype=hidden.dtype
            )
            # Stitch visual embeddings into text embedding stream at image pad positions
            image_mask = input_ids == self.config.image_token_id
            hidden[image_mask] = vision_out.to(dtype=hidden.dtype)

        for layer in self.layers:
            hidden = layer(
                hidden,
                position_ids=position_ids,
                attention_mask=attention_mask,
                metadata=metadata,
            )

        return self.norm(hidden)


class Qwen25VLForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen25VLModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen25VLTransformerModel(config)
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
        images: Optional[List[Any]] = None,
        image_grid_thw: Optional[List[Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            metadata=metadata,
            images=images,
            image_grid_thw=image_grid_thw,
        )
        return self.logits_processor(
            hidden_states,
            embed_tokens_weight=(
                self.model.embed_tokens.weight
                if self.config.tie_word_embeddings
                else None
            ),
        )
