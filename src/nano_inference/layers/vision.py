"""Vision encoder layers for Qwen2.5-VL and future VL models.

Design notes:
- VisionEncoderBase / VisionAttentionBase: ABCs that future VL models can sub-class.
- NaiveRMSNorm reused directly for ViT norms and PatchMerger.
- scaled_dot_product_attention (from attention.py) reused for the non-FA fallback.
- rotate_half (from rotary.py) reused inside _apply_rotary_pos_emb_vision.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from nano_inference.layers.attention import scaled_dot_product_attention
from nano_inference.layers.norm import NaiveRMSNorm
from nano_inference.layers.rotary import rotate_half

# ─── Abstract base classes ────────────────────────────────────────────────────


class VisionEncoderBase(nn.Module, ABC):
    """Contract for any vision backbone (Qwen2.5-VL, Qwen3-VL, LLaVA CLIP, …).

    forward() accepts pre-processed pixel patches and returns a flat sequence of
    visual token embeddings ready to be stitched into the LLM embedding stream.
    """

    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: Pre-processed patches as output by the HF image_processor.
                Shape: (total_patches, C * temporal_patch_size, patch_h, patch_w)
            image_grid_thw: Per-image (T, H_patches, W_patches) before spatial merge.
        Returns:
            (total_llm_tokens, llm_hidden_size) — one row per visual token in LLM order.
        """
        ...


class VisionAttentionBase(nn.Module, ABC):
    """Contract for ViT-internal attention.

    Signature differs from decoder attention: no batch dimension; sequence is
    packed; images/windows are separated via cu_seqlens.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor: ...


# ─── Qwen2.5-VL ViT building blocks ─────────────────────────────────────────


class Qwen25VLPatchEmbed(nn.Module):
    """3-D patch embedding via Conv2d with temporal frames merged into channels.

    The HF image_processor outputs pixel_values of shape:
        (num_patches, C * temporal_patch_size, patch_h, patch_w)

    A single Conv2d with in_channels = C * temporal_patch_size handles this.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels * temporal_patch_size,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_patches, C*temporal_ps, patch_h, patch_w)
        # After Conv2d: (num_patches, hidden, 1, 1) → (num_patches, hidden)
        return self.proj(x).flatten(1)


class Qwen25VLVisionRotaryEmbedding(nn.Module):
    """2D spatial frequency table for ViT rotary position embeddings.

    forward(seqlen) returns raw outer-product frequencies (seqlen, dim // 2).
    The caller indexes this table by (h_pos, w_pos) to build full patch embeddings.
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        # dim = head_dim // 2 (e.g. 40 for head_dim=80)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(seq, self.inv_freq)  # (seqlen, dim // 2)


class VisionSwiGLUMLP(nn.Module):
    """SwiGLU MLP for Qwen2.5-VL ViT blocks (gate_proj * silu, then down_proj)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D spatial RoPE to ViT query/key tensors.

    q/k shape: (seq_len, num_heads, head_dim)  — no batch dimension in ViT
    cos/sin:   (seq_len, head_dim)              — broadcast over heads

    Reuses rotate_half() from layers/rotary.py; the formula is identical to
    the LLM RoPE but operates on unbatched sequences.
    """
    cos = cos.unsqueeze(1).to(q.dtype)  # (seq_len, 1, head_dim)
    sin = sin.unsqueeze(1).to(q.dtype)
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


class Qwen25VLVisionAttention(VisionAttentionBase):
    """Bidirectional multi-head self-attention for Qwen2.5-VL ViT.

    Key differences from decoder attention:
    - Non-causal (no masking).
    - No batch dimension: the entire visual sequence is packed into (seq_len, hidden).
    - Uses cu_seqlens to process each image/window chunk independently via
      the non-FA fallback (reuses scaled_dot_product_attention with is_causal=False).
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_len = x.shape[0]
        cos, sin = position_embeddings

        # Project → (seq_len, 3, num_heads, head_dim), then split q/k/v
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(
            0
        )  # each (seq_len, num_heads, head_dim)

        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Process each image / window chunk independently (non-FA path)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_chunks = q.split(lengths, dim=0)
        k_chunks = k.split(lengths, dim=0)
        v_chunks = v.split(lengths, dim=0)

        out_parts = []
        for q_c, k_c, v_c in zip(q_chunks, k_chunks, v_chunks):
            # (chunk_len, H, D) → (1, H, chunk_len, D) for scaled_dot_product_attention
            q_c = q_c.transpose(0, 1).unsqueeze(0)
            k_c = k_c.transpose(0, 1).unsqueeze(0)
            v_c = v_c.transpose(0, 1).unsqueeze(0)
            attn = scaled_dot_product_attention(q_c, k_c, v_c, is_causal=False)
            # (1, H, chunk_len, D) → (chunk_len, H*D)
            out_parts.append(
                attn.squeeze(0).transpose(0, 1).reshape(-1, q.shape[1] * q.shape[2])
            )

        return self.proj(torch.cat(out_parts, dim=0))


class Qwen25VLVisionBlock(nn.Module):
    """Pre-norm ViT transformer block: norm → attn → residual, norm → mlp → residual.

    Reuses NaiveRMSNorm for both normalisation layers (no subclassing needed;
    it already satisfies RMSNormBase).
    """

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = NaiveRMSNorm(hidden_size)
        self.attn = Qwen25VLVisionAttention(hidden_size, num_heads)
        self.norm2 = NaiveRMSNorm(hidden_size)
        self.mlp = VisionSwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens, position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen25VLPatchMerger(nn.Module):
    """2×2 spatial patch merger + MLP projection to LLM hidden dimension.

    Groups spatial_merge_size² neighbouring original patches (from the
    window-reordered sequence) into a single LLM token, then projects from
    context_dim × 4 → out_dim via a 2-layer GELU MLP.

    Reuses NaiveRMSNorm for the input normalisation layer (ln_q).
    """

    def __init__(
        self,
        context_dim: int,  # vision hidden (1280 for 3B)
        out_dim: int,  # LLM hidden (2048 for 3B)
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        merged_dim = context_dim * spatial_merge_size**2  # 1280 × 4 = 5120
        self.merged_dim = merged_dim
        self.ln_q = NaiveRMSNorm(context_dim)  # reuse existing layer
        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, merged_dim),
            nn.GELU(),
            nn.Linear(merged_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (total_original_patches, context_dim)
        x = self.ln_q(x).view(-1, self.merged_dim)  # (total_llm_tokens, merged_dim)
        return self.mlp(x)  # (total_llm_tokens, out_dim)


class Qwen25VisionTransformer(VisionEncoderBase):
    """Full Qwen2.5-VL vision backbone.

    Processing pipeline:
    1. Patch embedding via Conv2d.
    2. 2D spatial RoPE position embeddings.
    3. Window permutation: reorder spatial_merge_unit-groups into attention windows.
       CRITICAL — weights were trained with this ordering; skipping it produces garbage.
    4. 32 ViT blocks alternating between full attention (at fullatt_block_indexes)
       and window attention (all other layers).
    5. PatchMerger: 4 original patches → 1 LLM token.
    6. Reverse permutation: restore spatial (raster-scan) order.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth: int,
        intermediate_size: int,
        out_hidden_size: int,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        window_size: int = 112,
        fullatt_block_indexes: Tuple[int, ...] = (7, 15, 23, 31),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.fullatt_block_indexes: Set[int] = set(fullatt_block_indexes)
        # Window size expressed in LLM patches (after spatial merge)
        self.vit_merger_window_size = window_size // spatial_merge_size // patch_size

        self.patch_embed = Qwen25VLPatchEmbed(
            hidden_size=hidden_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        # Vision RoPE uses dim = head_dim // 2 frequency components
        self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(dim=self.head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen25VLVisionBlock(hidden_size, num_heads, intermediate_size)
                for _ in range(depth)
            ]
        )
        self.merger = Qwen25VLPatchMerger(
            context_dim=hidden_size,
            out_dim=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _rot_pos_emb(
        self,
        grid_thw: List[Tuple[int, int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D spatial position embeddings for all patches.

        The HF code interleaves h/w positions to align with the spatial_merge_size
        grouping (2×2 blocks of patches form one LLM token). This permutation ensures
        that neighbouring patches in the merger are also neighbours in pos-emb space.

        Returns (cos, sin) each of shape (total_patches, head_dim).
        """
        pos_ids_list = []
        for t, h, w in grid_thw:
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
            sms = self.spatial_merge_size
            hpos = (
                hpos.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).flatten()
            )
            wpos = (
                wpos.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).flatten()
            )
            pos_ids_list.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids_list, dim=0)  # (total_patches, 2)
        max_grid = max(max(h, w) for _, h, w in grid_thw)
        freq_table = self.rotary_pos_emb(max_grid)  # (max_grid, head_dim // 4)
        # Index h and w → (total_patches, 2, head_dim//4), flatten → (total_patches, head_dim//2)
        patch_freqs = freq_table[pos_ids].flatten(1)
        # Duplicate to full head_dim: (total_patches, head_dim)
        emb = torch.cat([patch_freqs, patch_freqs], dim=-1).to(
            device=device, dtype=dtype
        )
        return emb.cos(), emb.sin()

    def _get_window_index(
        self, grid_thw: List[Tuple[int, int, int]], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the window permutation and cumulative window sequence lengths.

        Patches are grouped into vit_merger_window_size × vit_merger_window_size
        windows (in LLM-patch space). Padding positions (sentinel -100) are excluded
        via unique_consecutive so empty windows don't create zero-length cu_seqlen gaps.

        Returns:
            window_index:     (total_llm_groups,) — permutation of spatial_merge_unit groups
            cu_window_seqlens: (num_non_empty_windows+1,) — cumulative original patch counts
        """
        window_index: list = []
        cu_window_seqlens: list = [0]
        group_offset = 0
        ws = self.vit_merger_window_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_h = grid_h // self.spatial_merge_size
            llm_w = grid_w // self.spatial_merge_size
            total_llm = grid_t * llm_h * llm_w
            index = torch.arange(total_llm).reshape(grid_t, llm_h, llm_w)

            # % ws handles the case where llm_h is already divisible by ws (no pad)
            pad_h = (ws - llm_h % ws) % ws
            pad_w = (ws - llm_w % ws) % ws
            num_win_h = (llm_h + pad_h) // ws
            num_win_w = (llm_w + pad_w) // ws

            index_padded = F.pad(index.float(), (0, pad_w, 0, pad_h), value=-100).long()
            # Reshape into (grid_t, nwh, ws, nww, ws) then group windows
            index_padded = index_padded.reshape(grid_t, num_win_h, ws, num_win_w, ws)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t * num_win_h * num_win_w, ws, ws
            )
            seqlens = (index_padded != -100).sum(dim=[1, 2])  # (num_windows,)
            valid = index_padded.reshape(-1)
            window_index.append(valid[valid != -100] + group_offset)

            cum = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cum.tolist())
            group_offset += total_llm

        wi = torch.cat(window_index, dim=0).to(device)
        cws = torch.tensor(cu_window_seqlens, dtype=torch.int32, device=device)
        cws = torch.unique_consecutive(cws)
        return wi, cws

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: List[Tuple[int, int, int]],
    ) -> torch.Tensor:
        device = pixel_values.device
        dtype = pixel_values.dtype

        # 1. Patch embedding: (total_patches, C*tps, ph, pw) → (total_patches, hidden)
        hidden = self.patch_embed(pixel_values)

        # 2. 2D rotary position embeddings
        cos, sin = self._rot_pos_emb(image_grid_thw, device, dtype)

        # 3. Spatial reordering into windows (operates on spatial_merge_unit groups)
        window_index, cu_window_seqlens = self._get_window_index(image_grid_thw, device)
        seq_len = hidden.shape[0]
        smu = self.spatial_merge_unit

        def _reorder(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(seq_len // smu, smu, -1)[window_index].reshape(seq_len, -1)

        hidden = _reorder(hidden)
        cos = _reorder(cos)
        sin = _reorder(sin)
        position_embeddings = (cos, sin)

        # 4. Full-attention cu_seqlens: one boundary per image frame
        grid_t = torch.tensor(
            [t for t, _, _ in image_grid_thw], dtype=torch.int32, device=device
        )
        grid_hw = torch.tensor(
            [h * w for _, h, w in image_grid_thw], dtype=torch.int32, device=device
        )
        cu_seqlens = torch.repeat_interleave(grid_hw, grid_t).cumsum(
            0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # 5. ViT blocks
        for i, block in enumerate(self.blocks):
            cu_now = (
                cu_seqlens if i in self.fullatt_block_indexes else cu_window_seqlens
            )
            hidden = block(hidden, cu_now, position_embeddings)

        # 6. Patch merger: 4 original patches → 1 LLM token
        merged = self.merger(hidden)  # (total_llm_tokens, out_hidden_size)

        # 7. Reverse window permutation → spatial (raster-scan) order
        return merged[torch.argsort(window_index)]
