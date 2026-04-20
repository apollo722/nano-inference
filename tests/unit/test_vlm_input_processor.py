"""Unit tests for Qwen25VLInputProcessor — grid formula, mrope shape, token count."""

from unittest.mock import MagicMock

import pytest
import torch
from nano_inference.input_processor.vlm import Qwen25VLInputProcessor

# Qwen2.5-VL image pad token ID (used to construct test token sequences)
IMAGE_TOKEN_ID = Qwen25VLInputProcessor._FALLBACK_IMAGE_TOKEN_ID

IMAGE_H = 448
IMAGE_W = 448
# 448 → rounds to 448 (already multiple of 28) → H//14=32, W//14=32
# llm_h=16, llm_w=16 → visual_tokens = 1*16*16 = 256


def _make_fake_image(h=IMAGE_H, w=IMAGE_W):
    img = MagicMock()
    img.height = h
    img.width = w
    return img


def _make_processor(token_ids):
    """Return a minimal mock processor."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = token_ids
    proc = MagicMock()
    proc.apply_chat_template.return_value = "dummy text"
    proc.encode = tokenizer.encode
    return tokenizer, proc


def _make_tokenizer():
    """Return a mock tokenizer that returns IMAGE_TOKEN_ID for convert_tokens_to_ids."""
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.return_value = IMAGE_TOKEN_ID
    tokenizer.unk_token_id = 0
    return tokenizer


class TestComputeGrid:
    def setup_method(self):
        self.vp = Qwen25VLInputProcessor(
            tokenizer=_make_tokenizer(), processor=None, spatial_merge_size=2
        )

    def test_square_448(self):
        img = _make_fake_image(448, 448)
        t, h, w = self.vp._compute_grid(img)
        assert t == 1
        assert h == 32  # 448 // 14
        assert w == 32

    def test_round_up_to_28(self):
        # 300 / 28 ≈ 10.7 → rounds to 11 → 308; 308 // 14 = 22
        img = _make_fake_image(300, 300)
        t, h, w = self.vp._compute_grid(img)
        assert t == 1
        assert h == 22
        assert w == 22

    def test_minimum_one_tile(self):
        img = _make_fake_image(1, 1)
        t, h, w = self.vp._compute_grid(img)
        assert t == 1
        assert h >= 2  # at least 1 tile = 28px → 2 patches
        assert w >= 2

    def test_visual_token_count(self):
        img = _make_fake_image(448, 448)
        t, h, w = self.vp._compute_grid(img)
        visual_tokens = t * (h // 2) * (w // 2)
        assert visual_tokens == 256  # 1 * 16 * 16


class TestComputeMropePositions:
    def setup_method(self):
        self.vp = Qwen25VLInputProcessor(
            tokenizer=_make_tokenizer(), processor=None, spatial_merge_size=2
        )

    def _image_token_ids(self, count):
        return [IMAGE_TOKEN_ID] * count

    def test_text_only_positions(self):
        token_ids = [1, 2, 3, 4]
        mrope = self.vp._compute_mrope_positions(token_ids, [])
        assert mrope.shape == (3, 4)
        # All 3 axes equal (text tokens use same position)
        assert torch.all(mrope[0] == mrope[1])
        assert torch.all(mrope[1] == mrope[2])
        # Positions are [0, 1, 2, 3]
        assert mrope[0].tolist() == [0, 1, 2, 3]

    def test_image_plus_text_shape(self):
        # 1*16*16=256 image tokens + 4 text tokens
        token_ids = self._image_token_ids(256) + [1, 2, 3, 4]
        grid_thw = [(1, 32, 32)]  # T=1, H=32, W=32 → llm_h=16, llm_w=16
        mrope = self.vp._compute_mrope_positions(token_ids, grid_thw)
        assert mrope.shape == (3, 260)

    def test_image_position_advance(self):
        # After image: current_pos advances by max(llm_h, llm_w) = 16 (not 256)
        token_ids = self._image_token_ids(256) + [999]
        grid_thw = [(1, 32, 32)]
        mrope = self.vp._compute_mrope_positions(token_ids, grid_thw)
        # The text token after the image should have pos = 16 (not 256)
        text_temporal_pos = mrope[0, 256].item()
        text_height_pos = mrope[1, 256].item()
        text_width_pos = mrope[2, 256].item()
        assert text_temporal_pos == text_height_pos == text_width_pos
        assert text_temporal_pos == 16

    def test_mrope_dtype(self):
        token_ids = [1, 2, 3]
        mrope = self.vp._compute_mrope_positions(token_ids, [])
        assert mrope.dtype == torch.long
