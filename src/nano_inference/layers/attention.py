import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
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

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    head_dim = q.shape[-1]

    # Compute scaled similarity scores
    # (B, H, S, S) = (B, H, S, D) @ (B, H, D, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply causal mask to prevent attention to future tokens
    if is_causal:
        query_len = q.shape[-2]
        key_len = k.shape[-2]
        causal_mask = torch.triu(
            torch.ones(query_len, key_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Apply external attention mask (e.g., for padding)
    if attention_mask is not None:
        # attention_mask is expected to be already broadcastable to (B, H, S, S)
        # and already converted to additive form (0 for keep, -inf for mask)
        scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)

    # Normalize scores to probabilities along the last dimension
    attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    # Compute final attention output as weighted sum of values
    # (B, H, S, D) = (B, H, S, S) @ (B, H, S, D)
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
            # Convert boolean (True=keep, False=mask) to additive (0=keep, -inf=mask)
            mask_additive = torch.zeros_like(attention_mask, dtype=x.dtype)
            mask_additive = mask_additive.masked_fill(~attention_mask, float("-inf"))
            # Reshape for broadcasting with (B, H, S_step, S_total)
            # mask_additive is (B, S_total), we need (B, 1, 1, S_total)
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


from nano_inference.core.context import AttentionMetadata

...


class PagedCausalSelfAttention(NaiveCausalSelfAttention):
    """
    Paged Attention layer.

    Instead of recomputing full history or using a contiguous KV cache,
    this layer reads from and writes to a paged block-based storage.
    """

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Input Shapes:
            x: (batch, seq_step, hidden_size)
            position_ids: (batch, seq_step)
            attention_mask: (batch, seq_total)
            metadata: Contains block_tables, slot_mapping, context_lens, and caches.
        """
        if metadata is None:
            # Fallback to naive if metadata is not provided
            return super().forward(x, position_ids, attention_mask)

        batch_size, seq_step, _ = x.shape
        k_cache = metadata.k_cache
        v_cache = metadata.v_cache
        slot_mapping = metadata.slot_mapping
        kv_block_tables = metadata.kv_block_tables
        context_lens = metadata.context_lens

        # 1. Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_step, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_step, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_step, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # 2. Apply Rotary Positional Embeddings
        q, k = self.rotary(q, k, position_ids)

        # 3. Write Phase: Scatter current K/V into physical cache
        if k_cache is not None and v_cache is not None and slot_mapping is not None:
            flat_slot_mapping = slot_mapping.view(-1)
            num_kv_heads = self.num_kv_heads
            head_dim = self.head_dim
            block_size = k_cache.shape[2]

            k_cache_flat = k_cache.view(-1, num_kv_heads, head_dim)
            v_cache_flat = v_cache.view(-1, num_kv_heads, head_dim)

            k_cache_flat[flat_slot_mapping] = k.view(-1, num_kv_heads, head_dim)
            v_cache_flat[flat_slot_mapping] = v.view(-1, num_kv_heads, head_dim)

        # 4. Gather Phase: Reconstruct contiguous K/V for attention
        if (
            k_cache is not None
            and kv_block_tables is not None
            and context_lens is not None
        ):
            max_context_len = context_lens.max().item()
            num_kv_heads = self.num_kv_heads
            head_dim = self.head_dim
            block_size = k_cache.shape[2]

            # Vectorized Gather
            # 1. Build a (batch, max_context_len) index tensor of physical slot indices
            # Each entry is physical_block_id * block_size + block_offset

            # [B, max_context_len]
            token_indices = (
                torch.arange(max_context_len, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            # [B, max_context_len]
            block_indices = token_indices // block_size
            # [B, max_context_len]
            block_offsets = token_indices % block_size

            # Map logical block index to physical block ID using kv_block_tables [B, max_blocks]
            # [B, max_context_len]
            physical_block_ids = torch.gather(kv_block_tables.long(), 1, block_indices)

            # Final physical slot indices: [B, max_context_len]
            flat_indices = (physical_block_ids * block_size + block_offsets).view(-1)

            # 2. Gather from flattened cache
            # k_cache is [num_blocks, num_kv_heads, block_size, head_dim]
            # Flatten to [total_slots, num_kv_heads, head_dim]
            k_cache_flat = k_cache.view(-1, num_kv_heads, head_dim)
            v_cache_flat = v_cache.view(-1, num_kv_heads, head_dim)

            # Gather: [B * max_context_len, num_kv_heads, head_dim]
            k_contig = k_cache_flat[flat_indices].view(
                batch_size, max_context_len, num_kv_heads, head_dim
            )
            v_contig = v_cache_flat[flat_indices].view(
                batch_size, max_context_len, num_kv_heads, head_dim
            )

            # GQA: Repeat heads if necessary
            if self.num_kv_heads != self.num_heads:
                num_repeats = self.num_heads // self.num_kv_heads
                k_contig = k_contig.repeat_interleave(num_repeats, dim=2)
                v_contig = v_contig.repeat_interleave(num_repeats, dim=2)

            # Transpose for attention: [B, H, S, D]
            k_contig = k_contig.transpose(1, 2)
            v_contig = v_contig.transpose(1, 2)
            q = q.transpose(1, 2)
        else:
            return super().forward(x, position_ids, attention_mask)

        # 5. Attention Phase
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_additive = torch.zeros_like(attention_mask, dtype=x.dtype)
            mask_additive = mask_additive.masked_fill(~attention_mask, float("-inf"))
            attention_mask = mask_additive.view(batch_size, 1, 1, -1)

        attn_out = scaled_dot_product_attention(
            q=q,
            k=k_contig,
            v=v_contig,
            attention_mask=attention_mask,
            is_causal=True,
        )

        # 6. Output projection
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_step, self.num_heads * self.head_dim)
        )

        return self.o_proj(attn_out)
