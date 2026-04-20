"""Unit tests for the Qwen2.5-VL vision encoder (shape correctness, no weight loading)."""

import pytest
import torch
from nano_inference.layers.vision import (
    Qwen25VisionTransformer,
    Qwen25VLPatchEmbed,
    Qwen25VLPatchMerger,
)

# Qwen2.5-VL-3B vision config
VIT_CONFIG = dict(
    hidden_size=1280,
    num_heads=16,
    depth=4,  # Use 4 blocks instead of 32 for speed
    intermediate_size=3420,
    out_hidden_size=2048,
    patch_size=14,
    temporal_patch_size=2,
    spatial_merge_size=2,
    window_size=112,
    fullatt_block_indexes=(3,),  # Full attention on last block only
)


class TestQwen25VLPatchEmbed:
    def test_output_shape(self):
        embed = Qwen25VLPatchEmbed(
            in_channels=3, hidden_size=1280, patch_size=14, temporal_patch_size=2
        )
        # HF image_processor outputs (num_patches, C*temporal_ps, ph, pw)
        num_patches = 8
        x = torch.randn(num_patches, 3 * 2, 14, 14)
        out = embed(x)
        assert out.shape == (num_patches, 1280)


class TestQwen25VLPatchMerger:
    def test_output_shape(self):
        merger = Qwen25VLPatchMerger(
            context_dim=1280, out_dim=2048, spatial_merge_size=2
        )
        # merger expects (total_llm_tokens * 4, context_dim) — 4 patches per LLM token
        n_llm = 16
        x = torch.randn(n_llm * 4, 1280)
        out = merger(x)
        assert out.shape == (n_llm, 2048)


class TestQwen25VisionTransformer:
    @pytest.fixture(scope="class")
    def model(self):
        m = Qwen25VisionTransformer(**VIT_CONFIG)
        m.eval()
        return m

    def _make_pixel_values(self, grid_thw):
        """Create fake pixel_values matching a given grid_thw."""
        total_patches = sum(t * h * w for t, h, w in grid_thw)
        return torch.randn(total_patches, 3 * 2, 14, 14)

    def test_single_image_output_shape(self, model):
        grid_thw = [(1, 4, 4)]  # T=1, H=4, W=4 → 16 patches → 4 LLM tokens
        pixel_values = self._make_pixel_values(grid_thw)
        with torch.no_grad():
            out = model(pixel_values, grid_thw)
        expected_llm_tokens = 1 * (4 // 2) * (4 // 2)  # = 4
        assert out.shape == (expected_llm_tokens, 2048)

    def test_larger_image_output_shape(self, model):
        # 4x4 LLM grid (need 8×8 patches before merge = 32/32)
        grid_thw = [(1, 8, 8)]
        pixel_values = self._make_pixel_values(grid_thw)
        with torch.no_grad():
            out = model(pixel_values, grid_thw)
        expected_llm_tokens = 1 * 4 * 4  # 4*4 LLM tiles
        assert out.shape == (expected_llm_tokens, 2048)

    def test_two_images_batch(self, model):
        grid_thw = [(1, 4, 4), (1, 4, 4)]
        pixel_values = self._make_pixel_values(grid_thw)
        with torch.no_grad():
            out = model(pixel_values, grid_thw)
        expected_llm_tokens = 4 + 4  # 4 LLM tokens per image
        assert out.shape == (expected_llm_tokens, 2048)

    def test_output_dtype_preserved(self, model):
        grid_thw = [(1, 4, 4)]
        pixel_values = self._make_pixel_values(grid_thw).to(torch.float32)
        with torch.no_grad():
            out = model(pixel_values, grid_thw)
        assert out.dtype == torch.float32
