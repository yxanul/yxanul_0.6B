#!/usr/bin/env python3
"""
GPT model (v2) using TransformerEngine FP8 with cuDNN DotProductAttention (DPA).

- Attention runs via TE's cuDNN DPA path to enable FP8 attention and native GQA
  (no KV duplication via repeat_interleave).
- Uses FP8 DelayedScaling recipe; BF16 parameters for projections and norms.

This variant focuses on: FP8 attention + fused QKV projection for fewer launches.
"""

import os
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Prefer cuDNN/DPA attention (do NOT force flash-attn)
# If user already set this, we will not override it.
os.environ.setdefault("NVTE_FUSED_ATTN_BACKEND", "0")

# TransformerEngine imports
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    try:
        # TE >= 2.5
        from transformer_engine.pytorch.attention import (
            DotProductAttention as TE_DPA,
        )
    except Exception:
        TE_DPA = None
    TE_AVAILABLE = True
except ImportError:
    print("ERROR: TransformerEngine not available!")
    import sys
    sys.exit(1)


@dataclass
class ModelConfig:
    vocab_size: int = 49152
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_heads: int = 3
    block_size: int = 2048
    dropout: float = 0.05
    bias: bool = False
    rope_theta: float = 10000.0
    # FP8 configuration
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    # Compatibility knob kept (no-op here)
    fuse_wgrad_accumulation: bool = False

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_heads == 0
        self.head_dim = self.n_embd // self.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        # Align for FP8 kernels
        assert self.n_embd % 16 == 0, "n_embd must be divisible by 16 for FP8"
        assert self.head_dim % 16 == 0, "head_dim must be divisible by 16 for FP8"


def get_fp8_recipe(config: ModelConfig):
    """Get the FP8 recipe for training."""
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    )
    print(f"GPU: {device_name}")
    print("Using DelayedScaling FP8 (v2, cuDNN DPA attention)")
    print("  Note: Using TE cuDNN attention path for FP8 & native GQA")

    return DelayedScaling(
        fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
        amax_history_len=config.fp8_amax_history_len,
        amax_compute_algo=config.fp8_amax_compute_algo,
    )


class RoPE:
    """Rotary Position Embeddings."""

    @staticmethod
    def create_cos_sin_cache(seq_len, n_elem, base=10000, device="cpu"):
        theta = 1.0 / (
            base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem)
        )
        seq_idx = torch.arange(seq_len, device=device).float()
        idx_theta = torch.outer(seq_idx, theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        return cache

    @staticmethod
    def apply_rotary_pos_emb(x, cos_sin_cache):
        batch, seq_len, n_heads, head_dim = x.shape
        x = x.reshape(batch, seq_len, n_heads, head_dim // 2, 2)
        cos_cache = cos_sin_cache[:seq_len, :, 0].to(device=x.device, dtype=x.dtype)
        sin_cache = cos_sin_cache[:seq_len, :, 1].to(device=x.device, dtype=x.dtype)

        x_rot = torch.stack(
            [
                x[..., 0] * cos_cache.unsqueeze(0).unsqueeze(2)
                - x[..., 1] * sin_cache.unsqueeze(0).unsqueeze(2),
                x[..., 0] * sin_cache.unsqueeze(0).unsqueeze(2)
                + x[..., 1] * cos_cache.unsqueeze(0).unsqueeze(2),
            ],
            dim=-1,
        )

        return x_rot.reshape(batch, seq_len, n_heads, head_dim)


class CleanAttention(nn.Module):
    """Multi-Head Attention with native GQA via TE cuDNN DPA (FP8).

    - Uses fused QKV projection for fewer launches.
    - If TE DPA is unavailable, falls back to PyTorch SDPA (BF16) and
      repeats KV heads to match n_head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim

        # Fused QKV: [B, T, (H + 2*KV)*D]
        self.qkv = te.Linear(
            config.n_embd,
            (self.n_head + 2 * self.n_kv_heads) * self.head_dim,
            bias=config.bias,
            params_dtype=torch.bfloat16,
        )
        self.o_proj = te.Linear(
            self.n_head * self.head_dim,
            config.n_embd,
            bias=config.bias,
            params_dtype=torch.bfloat16,
        )

        if TE_DPA is not None:
            # TE cuDNN attention handles GQA and attention dropout internally
            self.dpa = TE_DPA(
                num_attention_heads=self.n_head,
                num_kv_heads=self.n_kv_heads,
                attention_dropout=config.dropout,
                attn_mask_type="causal",
            )
            self._use_te_dpa = True
        else:
            # Fallback path: PyTorch SDPA (BF16), replicate KV
            self._use_te_dpa = False
            self._fallback_dropout = nn.Dropout(config.dropout)
            self._fallback_dropout_p = config.dropout

    def forward(self, x, rope_cache):
        B, T, C = x.shape

        # Project to QKV and reshape to [B, T, groups, D]
        qkv = self.qkv(x).view(
            B, T, (self.n_head + 2 * self.n_kv_heads), self.head_dim
        )
        q, k, v = torch.split(
            qkv, [self.n_head, self.n_kv_heads, self.n_kv_heads], dim=2
        )

        # Apply RoPE on Q and K (shape [B, T, H, D] / [B, T, KV, D])
        q = RoPE.apply_rotary_pos_emb(q, rope_cache)
        k = RoPE.apply_rotary_pos_emb(k, rope_cache)

        if self._use_te_dpa:
            # TE DPA expects [B, H, T, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            # Under te.fp8_autocast, this executes in FP8 where supported
            y = self.dpa(q, k, v)
            # Back to [B, T, H, D]
            y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
            # Output projection (no extra dropout here; DPA applied attention dropout)
            y = self.o_proj(y)
            return y
        else:
            # Fallback: PyTorch SDPA with KV head replication (BF16)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.n_kv_heads != self.n_head:
                k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
                v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0 if not self.training else self._fallback_dropout_p,
                is_causal=True,
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.o_proj(y)
            y = self._fallback_dropout(y)
            return y


class CleanFeedForward(nn.Module):
    """SwiGLU feedforward"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(config.n_embd * 8 / 3)
        hidden_dim = (hidden_dim + 63) // 64 * 64

        self.gate_proj = te.Linear(
            config.n_embd, hidden_dim, bias=config.bias, params_dtype=torch.bfloat16
        )
        self.up_proj = te.Linear(
            config.n_embd, hidden_dim, bias=config.bias, params_dtype=torch.bfloat16
        )
        self.down_proj = te.Linear(
            hidden_dim, config.n_embd, bias=config.bias, params_dtype=torch.bfloat16
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class CleanBlock(nn.Module):
    """Transformer block"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = CleanAttention(config)
        self.ln_2 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.ffn = CleanFeedForward(config)

    def forward(self, x, rope_cache):
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = x + self.ffn(self.ln_2(x))
        return x


class CleanGPT_TE(nn.Module):
    """GPT model with TransformerEngine FP8 and cuDNN DPA attention (v2)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings (BF16)
        wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Transformer blocks
        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([CleanBlock(config) for _ in range(config.n_layer)]),
                ln_f=te.RMSNorm(config.n_embd, eps=1e-6),
            )
        )

        # RoPE cache
        self.register_buffer(
            "rope_cache",
            RoPE.create_cos_sin_cache(
                config.block_size, config.head_dim, config.rope_theta
            ),
        )

        # Initialize weights
        self.apply(self._init_weights)
        # Tie after init to avoid double-initialization RNG effects
        self.lm_head.weight = self.transformer.wte.weight

        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # Report configuration
        total_params = self.num_parameters()
        print("v2 model initialized with TE FP8 + cuDNN DPA attention")
        print(
            f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)"
        )
        print(
            f"  Arch: {config.n_layer}L, {config.n_head}H, {config.n_embd}D | GQA: {config.n_kv_heads}"
        )
        print("  Attention backend: TE cuDNN DPA (FP8 where enabled); fallback SDPA if unavailable")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            if module is self.lm_head:
                # Will be tied to wte afterward
                return
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Sequence length {t} exceeds block size {self.config.block_size}"

        # Token embeddings
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        # Blocks
        for block in self.transformer.h:
            x = block(x, self.rope_cache)

        # Final norm and head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. FP8 requires GPU.")
        import sys
        sys.exit(0)

