"""VLM input processor base class and Qwen2.5-VL concrete implementation.

CPU-side determinism: all visual token counting and mRoPE position computation
happens here, before the scheduler runs, so KV block allocation needs no GPU sync.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
from nano_inference.core.request import GenerationInputs
from nano_inference.input_processor.base import (
    BaseInputProcessor,
    register_input_processor,
)


class VLMInputProcessorBase(BaseInputProcessor, ABC):
    """Abstract base for vision-language input processors.

    Subclasses must:
    - Set IMAGE_TOKEN_NAME to the placeholder token string in the vocabulary.
    - Implement encode() with model-specific tokenization and metadata computation.

    The image_token_id is resolved from the live tokenizer at construction time so
    callers never need to hardcode token IDs. Subclasses should supply a fallback
    default via _default_image_token_id() for environments where the tokenizer
    does not surface the token (e.g. unit-test mocks).
    """

    IMAGE_TOKEN_NAME: ClassVar[str] = ""

    def __init__(self, tokenizer, processor=None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.processor = processor
        self.image_token_id: int = self._resolve_image_token_id(tokenizer)

    def _resolve_image_token_id(self, tokenizer) -> int:
        """Resolve image placeholder token ID from the tokenizer vocabulary.

        Falls back to _default_image_token_id() when the tokenizer does not
        return a valid integer (e.g. unit-test mocks).
        """
        if self.IMAGE_TOKEN_NAME:
            tok_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN_NAME)
            unk = getattr(tokenizer, "unk_token_id", None)
            if isinstance(tok_id, int) and tok_id != unk:
                return tok_id
        return self._default_image_token_id()

    def _default_image_token_id(self) -> int:
        """Return a safe fallback image token ID. Override in each subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} must set IMAGE_TOKEN_NAME or override "
            "_default_image_token_id()"
        )

    @abstractmethod
    def encode(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Any]] = None,
        **kwargs,
    ) -> GenerationInputs: ...


@register_input_processor
class Qwen25VLInputProcessor(VLMInputProcessorBase):
    """Qwen2.5-VL input processor.

    Responsibilities (CPU-only):
    - Chat template formatting via HF AutoProcessor.
    - Image grid computation (T, H_patches, W_patches).
    - mRoPE position ID construction for the full prompt sequence.
    """

    name = "qwen25_vl"
    IMAGE_TOKEN_NAME: ClassVar[str] = "<|image_pad|>"
    _FALLBACK_IMAGE_TOKEN_ID: ClassVar[int] = 151655  # Qwen2.5-VL default

    def __init__(
        self, tokenizer, processor=None, spatial_merge_size: int = 2, **kwargs
    ):
        super().__init__(tokenizer, processor=processor, **kwargs)
        self.spatial_merge_size = spatial_merge_size

    def _default_image_token_id(self) -> int:
        return self._FALLBACK_IMAGE_TOKEN_ID

    # ── public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Any]] = None,
        max_prompt_tokens: Optional[int] = None,
        **kwargs,
    ) -> GenerationInputs:
        """Tokenize messages and compute all VLM-specific metadata on CPU.

        Args:
            messages: OpenAI-style chat messages (may contain image content parts).
            images: PIL Image objects, one per image reference in messages.

        Returns:
            GenerationInputs with prompt_token_ids, images, image_grid_thw,
            and mrope_position_ids already computed.
        """
        proc = self.processor or self.tokenizer

        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if not images:
            token_ids: List[int] = self.tokenizer.encode(text)
            return GenerationInputs(prompt_token_ids=token_ids)

        # Use the full HF processor call to get correctly expanded token IDs.
        # apply_chat_template alone inserts only 1 <|image_pad|> per image;
        # the processor expands it to T*H_llm*W_llm based on actual image size.
        hf_inputs = proc(
            text=text,
            images=images,
            return_tensors="pt",
            padding=False,
        )
        token_ids = hf_inputs.input_ids[0].tolist()

        # Extract image_grid_thw: (num_images, 3) tensor → list of (T, H, W) tuples
        raw_thw = hf_inputs.image_grid_thw
        if raw_thw is not None and hasattr(raw_thw, "tolist"):
            image_grid_thw = [tuple(int(v) for v in row) for row in raw_thw.tolist()]
        else:
            image_grid_thw = [self._compute_grid(img) for img in images]

        mrope_position_ids = self._compute_mrope_positions(token_ids, image_grid_thw)

        return GenerationInputs(
            prompt_token_ids=token_ids,
            images=images,
            image_grid_thw=image_grid_thw,
            mrope_position_ids=mrope_position_ids,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _compute_grid(self, image: Any) -> Tuple[int, int, int]:
        """Compute (T, H_patches, W_patches) for an image before spatial merge.

        Dimensions are rounded to the nearest multiple of patch_size * spatial_merge_size
        (= 28) to ensure divisibility by both the patch stride and the 2×2 merger.
        """
        patch_size = 14
        align = patch_size * self.spatial_merge_size  # 28
        H = max(round(image.height / align) * align, align)
        W = max(round(image.width / align) * align, align)
        return (1, H // patch_size, W // patch_size)

    def _compute_mrope_positions(
        self,
        token_ids: List[int],
        image_grid_thw: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """Compute 3-axis mRoPE position IDs for the full prompt sequence.

        Algorithm (mirrors HF get_rope_index for images):
        - Walk token_ids left to right tracking current_pos (1-D counter).
        - Text tokens: position = (cur, cur, cur) — same as 1D RoPE.
        - Image token runs: assign 3D grid positions (temporal, h, w) starting at
          current_pos; afterwards advance current_pos by max(llm_h, llm_w).

        The mrope_position_delta ensures subsequent text tokens start at the
        spatial extent of the image rather than at visual_token_count.

        Returns: torch.LongTensor of shape (3, seq_len)
        """
        sms = self.spatial_merge_size
        pos_t_list: List[int] = []
        pos_h_list: List[int] = []
        pos_w_list: List[int] = []

        current_pos = 0
        image_iter = iter(image_grid_thw)
        i = 0
        n = len(token_ids)

        while i < n:
            if token_ids[i] != self.image_token_id:
                pos_t_list.append(current_pos)
                pos_h_list.append(current_pos)
                pos_w_list.append(current_pos)
                current_pos += 1
                i += 1
            else:
                run_start = i
                while i < n and token_ids[i] == self.image_token_id:
                    i += 1
                run_len = i - run_start

                grid_t, grid_h, grid_w = next(image_iter)
                llm_t = grid_t
                llm_h = grid_h // sms
                llm_w = grid_w // sms

                t_ids, h_ids, w_ids = self._image_position_ids(
                    start=current_pos,
                    llm_t=llm_t,
                    llm_h=llm_h,
                    llm_w=llm_w,
                )
                assert len(t_ids) == run_len, (
                    f"Token count mismatch: image grid gives {len(t_ids)} tokens "
                    f"but found {run_len} <|image_pad|> tokens in prompt. "
                    f"grid=(T={grid_t}, H={grid_h}, W={grid_w}), "
                    f"llm_grid=(T={llm_t}, H={llm_h}, W={llm_w})"
                )
                pos_t_list.extend(t_ids)
                pos_h_list.extend(h_ids)
                pos_w_list.extend(w_ids)
                current_pos += max(llm_h, llm_w)

        return torch.tensor([pos_t_list, pos_h_list, pos_w_list], dtype=torch.long)

    @staticmethod
    def _image_position_ids(
        start: int, llm_t: int, llm_h: int, llm_w: int
    ) -> Tuple[List[int], List[int], List[int]]:
        """Generate (temporal, height, width) position ID lists for one image block."""
        t_ids: List[int] = []
        h_ids: List[int] = []
        w_ids: List[int] = []
        for _t in range(llm_t):
            for h in range(llm_h):
                for w in range(llm_w):
                    t_ids.append(start)
                    h_ids.append(start + h)
                    w_ids.append(start + w)
        return t_ids, h_ids, w_ids
