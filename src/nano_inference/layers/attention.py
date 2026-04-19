import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from nano_inference.core.context import AttentionMetadata
from nano_inference.layers.norm import NaiveRMSNorm
from nano_inference.layers.rotary import NaiveRotaryEmbedding


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention.
    """
    head_dim = q.shape[-1]
    query_len = q.shape[-2]
    key_len = k.shape[-2]

    # Compute scaled similarity scores (B, H, S_q, S_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply causal mask
    # If is_causal is True and it's a square matrix (prefill), apply triu.
    # If it's a decode step (query_len=1, key_len>1), it's naturally causal
    # as long as we don't have future keys in the cache (which we don't).
    if is_causal and query_len == key_len:
        causal_mask = torch.triu(
            torch.ones(query_len, key_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Apply external attention mask (e.g., for padding or paged context)
    if attention_mask is not None:
        # attention_mask shape: (B, 1, 1, S_total)
        # scores shape: (B, H, S_q, S_k)
        # In mixed batches, S_k for this specific request might be smaller than S_total
        if attention_mask.shape[-1] > key_len:
            attention_mask = attention_mask[..., :key_len]
        scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)

    attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(attn_weights, v)


class CausalSelfAttentionBase(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_qk_norm: bool = False,
        qkv_bias: bool = False,
        out_bias: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass


class NaiveCausalSelfAttention(CausalSelfAttentionBase):
    """
    Causal Self-Attention layer supporting Grouped-Query Attention (GQA).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_qk_norm: bool = False,
        qkv_bias: bool = False,
        out_bias: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            use_qk_norm,
            qkv_bias,
            out_bias,
            rope_base,
        )

        if head_dim is None and hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, got {hidden_size=} and {num_heads=}"
            )

        if num_kv_heads is None:
            num_kv_heads = num_heads

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads must be divisible by num_kv_heads, got {num_heads=} and {num_kv_heads=}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.use_qk_norm = use_qk_norm

        # Projections: Input -> Q, K, V
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias
        )
        # Output projection back to model dimension
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=out_bias,
        )

        # Optional QK-Normalization (used in Qwen3)
        self.q_norm = NaiveRMSNorm(self.head_dim) if use_qk_norm else None
        self.k_norm = NaiveRMSNorm(self.head_dim) if use_qk_norm else None
        self.rotary = NaiveRotaryEmbedding(head_dim=self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[
            AttentionMetadata
        ] = None,  # Unused here but kept for compatibility
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = self.rotary(q, k, position_ids)

        if self.num_kv_heads != self.num_heads:
            num_repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_repeats, dim=2)
            v = v.repeat_interleave(num_repeats, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attention_mask is not None and attention_mask.dim() == 2:
            mask_additive = torch.zeros_like(attention_mask, dtype=x.dtype)
            mask_additive = mask_additive.masked_fill(~attention_mask, float("-inf"))
            attention_mask = mask_additive.view(batch_size, 1, 1, -1)

        attn_out = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            attention_mask=attention_mask,
            is_causal=True,
        )

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )

        return self.o_proj(attn_out)


class PagedCausalSelfAttention(NaiveCausalSelfAttention):
    """
    Paged Attention layer.

    Architecture:
    1. Writes new K/V tokens to physical blocks in the cache.
    2. Gathers historical K/V tokens from physical blocks into a contiguous tensor.
    3. Performs attention using the gathered K/V.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_idx: Optional[int] = None

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Paged Attention.
        """
        if metadata is None:
            return super().forward(x, position_ids, attention_mask)

        batch_size, seq_step, _ = x.shape

        # In multi-layer KV cache, metadata.k_cache has shape [L, num_blocks, H, block_size, D]
        assert (
            self.layer_idx is not None
        ), "layer_idx must be set for PagedCausalSelfAttention"

        k_cache_full = metadata.k_cache
        v_cache_full = metadata.v_cache

        if k_cache_full is not None and v_cache_full is not None:
            k_cache = k_cache_full[self.layer_idx]
            v_cache = v_cache_full[self.layer_idx]
        else:
            k_cache = None
            v_cache = None

        slot_mapping = metadata.slot_mapping
        kv_block_tables = metadata.kv_block_tables
        context_lens = metadata.context_lens

        # 1. Project and Norm
        q = self.q_proj(x).view(batch_size, seq_step, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_step, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_step, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # 2. RoPE
        q, k = self.rotary(q, k, position_ids)

        # 3. Write Phase (In-place update to physical cache)
        if k_cache is not None and v_cache is not None and slot_mapping is not None:
            block_size = k_cache.shape[2]
            for b in range(batch_size):
                for s in range(seq_step):
                    slot = slot_mapping[b, s].item()
                    if slot < 0:
                        continue
                    block_idx = slot // block_size
                    offset = slot % block_size
                    k_cache[block_idx, :, offset, :] = k[b, s]
                    v_cache[block_idx, :, offset, :] = v[b, s]

        # 4. Attention Phase Selection
        if metadata.is_prefill:
            # PREFILL: Use fresh K/V directly
            k_contig = k
            v_contig = v
        elif (
            k_cache is not None
            and kv_block_tables is not None
            and context_lens is not None
        ):
            # DECODE: Gather Phase (Reconstruct contiguous K/V from cache)
            max_context_len = context_lens.max().item()
            block_size = k_cache.shape[2]

            # Vectorized gather using physical slot indices
            token_indices = (
                torch.arange(max_context_len, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            block_indices = token_indices // block_size
            block_offsets = token_indices % block_size
            physical_block_ids = torch.gather(kv_block_tables.long(), 1, block_indices)

            k_contig = k_cache[physical_block_ids, :, block_offsets, :]
            v_contig = v_cache[physical_block_ids, :, block_offsets, :]
        else:
            # Fallback if no cache metadata provided
            return super().forward(x, position_ids, attention_mask)

        # 5. GQA Repeat (Repeat KV heads to match Q heads if using GQA)
        if self.num_kv_heads != self.num_heads:
            num_repeats = self.num_heads // self.num_kv_heads
            k_contig = k_contig.repeat_interleave(num_repeats, dim=2)
            v_contig = v_contig.repeat_interleave(num_repeats, dim=2)

        # 6. Transpose to [B, H, S, D] for attention kernels
        q = q.transpose(1, 2)
        k_contig = k_contig.transpose(1, 2)
        v_contig = v_contig.transpose(1, 2)

        # 7. Mask handling (Convert boolean mask to additive mask)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_additive = torch.zeros_like(attention_mask, dtype=x.dtype)
            mask_additive = mask_additive.masked_fill(~attention_mask, float("-inf"))
            # Expand to [B, 1, 1, S_total]
            attention_mask = mask_additive.view(batch_size, 1, 1, -1)

        # 8. Attention (Causal mask applies triu only when Q_len == K_len)
        # For DECODE, Q_len is typically 1 while K_len is the full history.
        attn_out = scaled_dot_product_attention(
            q=q, k=k_contig, v=v_contig, attention_mask=attention_mask, is_causal=True
        )

        # 9. Output projection back to model hidden size
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_step, self.num_heads * self.head_dim)
        )
        return self.o_proj(attn_out)
