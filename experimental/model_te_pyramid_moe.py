#!/usr/bin/env python3
"""Variant of model_te_optimized with Full Attention + Pyramid-Residual MoE layers (FP8-ready)."""

import os
os.environ.setdefault('NVTE_FUSED_ATTN_BACKEND', '1')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, List

# TransformerEngine imports
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
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
    # Use full attention (no GQA) when pairing with MoE; we still keep n_kv_heads for compatibility
    n_kv_heads: int = 12
    block_size: int = 2048
    dropout: float = 0.05
    bias: bool = False
    rope_theta: float = 10000.0
    # FP8 configuration
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    # Advanced optimizations
    fuse_wgrad_accumulation: bool = True
    cache_fp8_weights: bool = True
    
    # MoE configuration (Pyramid-Residual)
    use_moe: bool = True
    moe_num_experts: int = 8
    moe_top_k: int = 2              # top-k soft routing (set 1 for softmax-all routing)
    moe_min_expansion: float = 2.0  # min hidden expansion (x d_model)
    moe_max_expansion: float = 8.0  # max hidden expansion (x d_model)
    moe_router_dropout: float = 0.0
    moe_output_dropout: float = 0.0
    moe_layers: Optional[List[int]] = None   # which layers (0-indexed) use MoE; None => all layers
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head > 0
        self.head_dim = self.n_embd // self.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.n_embd % 16 == 0, "n_embd must be divisible by 16 for FP8"
        assert self.head_dim % 16 == 0, "head_dim must be divisible by 16 for FP8"

def get_fp8_recipe(config):
    """Get the FP8 recipe for training."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    print(f"GPU: {device_name}")
    print("Using DelayedScaling FP8 with advanced optimizations")
    
    return DelayedScaling(
        fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
        amax_history_len=config.fp8_amax_history_len,
        amax_compute_algo=config.fp8_amax_compute_algo,
    )


class RoPE:
    """Rotary Position Embeddings."""
    @staticmethod
    def create_cos_sin_cache(seq_len, n_elem, base=10000, device='cpu'):
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
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
        
        x_rot = torch.stack([
            x[..., 0] * cos_cache.unsqueeze(0).unsqueeze(2) - x[..., 1] * sin_cache.unsqueeze(0).unsqueeze(2),
            x[..., 0] * sin_cache.unsqueeze(0).unsqueeze(2) + x[..., 1] * cos_cache.unsqueeze(0).unsqueeze(2)
        ], dim=-1)
        
        return x_rot.reshape(batch, seq_len, n_heads, head_dim)


class OptimizedAttention(nn.Module):
    """Multi-Head Attention with GQA and gradient accumulation fusion."""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.fuse_wgrad = config.fuse_wgrad_accumulation
        
        # Fused QKV projection for gradient accumulation optimization
        if self.fuse_wgrad:
            self.qkv_proj = te.Linear(
                config.n_embd, 
                (config.n_head + 2 * config.n_kv_heads) * config.head_dim,
                bias=config.bias, 
                params_dtype=torch.bfloat16
            )
        else:
            self.q_proj = te.Linear(config.n_embd, config.n_head * config.head_dim, bias=config.bias, params_dtype=torch.bfloat16)
            self.k_proj = te.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=config.bias, params_dtype=torch.bfloat16)
            self.v_proj = te.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=config.bias, params_dtype=torch.bfloat16)
        
        self.o_proj = te.Linear(
            config.n_head * config.head_dim, 
            config.n_embd,
            bias=config.bias, 
            params_dtype=torch.bfloat16
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout
    
    def forward(self, x, rope_cache):
        B, T, C = x.shape
        
        if self.fuse_wgrad:
            qkv = self.qkv_proj(x)
            q, k, v = torch.split(qkv, [self.n_head * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        
        # Reshape to [B, n_*, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q = RoPE.apply_rotary_pos_emb(q.transpose(1, 2), rope_cache).transpose(1, 2)
        k = RoPE.apply_rotary_pos_emb(k.transpose(1, 2), rope_cache).transpose(1, 2)
        
        # GQA: Repeat KV heads
        if self.n_kv_heads != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        
        # SDPA
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.dropout_p,
            is_causal=True
        )
        
        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.o_proj(y)
        y = self.dropout(y)
        
        return y


class FullAttention(OptimizedAttention):
    """Identical to OptimizedAttention but forces full attention (no GQA).
    We simply override n_kv_heads to equal n_head.
    """
    def __init__(self, config):
        import copy
        cfg = copy.copy(config)
        cfg.n_kv_heads = cfg.n_head
        super().__init__(cfg)

class PyramidResidualMoE(nn.Module):
    """
    Pyramid-Residual Mixture-of-Experts (MoE) MLP with SwiGLU experts.
    
    - Experts have *pyramid* widths between [moe_min_expansion, moe_max_expansion] * d_model,
      spaced geometrically.
    - Router computes token-wise gating, optionally selects top-k experts.
    - Outputs are mixed and then added residually: y = x + Dropout( Σ w_e * Expert_e(x) ).
    - Uses TransformerEngine Linear layers to play nice with FP8 autocast.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.moe_num_experts
        self.top_k = max(1, int(config.moe_top_k))
        self.router = te.Linear(self.n_embd, self.num_experts, bias=config.bias, params_dtype=torch.bfloat16)
        self.router_dropout = nn.Dropout(config.moe_router_dropout)
        self.output_dropout = nn.Dropout(config.moe_output_dropout)
        
        # Geometric sequence of expansion ratios
        min_e = config.moe_min_expansion
        max_e = max(config.moe_min_expansion, config.moe_max_expansion)
        if self.num_experts == 1:
            ratios = [min_e]
        else:
            ratios = torch.logspace(
                start=math.log10(min_e),
                end=math.log10(max_e),
                steps=self.num_experts
            ).tolist()
        
        # Build experts: each expert is a SwiGLU MLP with its own width
        self.experts = nn.ModuleList()
        for r in ratios:
            hidden = int(self.n_embd * (8.0/3.0) * (r / (8.0/3.0)))  # scale around baseline 8/3
            hidden = (hidden + 63) // 64 * 64  # align for TE
            expert = nn.ModuleDict({
                "gate_proj": te.Linear(self.n_embd, hidden, bias=config.bias, params_dtype=torch.bfloat16),
                "up_proj":   te.Linear(self.n_embd, hidden, bias=config.bias, params_dtype=torch.bfloat16),
                "down_proj": te.Linear(hidden, self.n_embd, bias=config.bias, params_dtype=torch.bfloat16),
            })
            self.experts.append(expert)
        
        # Init
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        if config.fuse_wgrad_accumulation:
            self.init_main_grad()
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def init_main_grad(self):
        for p in self.parameters():
            if p.requires_grad:
                p.main_grad = torch.zeros_like(p, dtype=torch.float32, device=p.device)
    
    def forward(self, x):
        B, T, C = x.shape
        # Router: logits -> (optional) dropout -> softmax
        logits = self.router(x)
        if self.training and self.router_dropout.p > 0:
            logits = self.router_dropout(logits)
        gate = F.softmax(logits, dim=-1)  # [B,T,E]

        # Top-k sparse dispatch
        topk_val, topk_idx = torch.topk(gate, k=self.top_k, dim=-1)  # [B,T,k]
        x_flat = x.reshape(B * T, C)
        out_flat = torch.zeros_like(x_flat)

        assign = topk_idx.reshape(-1, self.top_k)    # [BT, k]
        weights = topk_val.reshape(-1, self.top_k)   # [BT, k]

        # Process only the tokens routed to each expert
        for e, expert in enumerate(self.experts):
            mask = (assign == e)
            if not mask.any():
                continue
            token_pos, which_k = mask.nonzero(as_tuple=False).unbind(1)  # [M]
            w = weights[token_pos, which_k].unsqueeze(1)                 # [M,1]
            tokens = x_flat.index_select(0, token_pos)                   # [M,C]

            g = expert["gate_proj"](tokens)
            u = expert["up_proj"](tokens)
            h = F.silu(g) * u
            h = expert["down_proj"](h)                                   # [M,C]

            out_flat.index_add_(0, token_pos, w * h)

        mixed = out_flat.view(B, T, C)
        out = x + self.output_dropout(mixed)
        return out

class DenseFFN(nn.Module):
    """Dense SwiGLU FFN used in non-MoE layers (TE-friendly)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embd = config.n_embd
        hidden = int(self.n_embd * (8.0/3.0))
        hidden = (hidden + 63) // 64 * 64
        self.gate_proj = te.Linear(self.n_embd, hidden, bias=config.bias, params_dtype=torch.bfloat16)
        self.up_proj   = te.Linear(self.n_embd, hidden, bias=config.bias, params_dtype=torch.bfloat16)
        self.down_proj = te.Linear(hidden, self.n_embd, bias=config.bias, params_dtype=torch.bfloat16)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = F.silu(g) * u
        h = self.down_proj(h)
        return x + self.dropout(h)

class FullAttnFFNBlock(nn.Module):
    """Transformer block: Full self-attention + Dense FFN."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = FullAttention(config)
        self.ln_2 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.ffn  = DenseFFN(config)

    def forward(self, x, rope_cache):
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = self.ffn(self.ln_2(x))
        return x

class FullAttnMoEBlock(nn.Module):
    """Transformer block: Full self-attention + Pyramid-Residual MoE MLP."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = FullAttention(config)
        self.ln_2 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.moe  = PyramidResidualMoE(config)
    
    def forward(self, x, rope_cache):
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = self.moe(self.ln_2(x))  # residual inside MoE
        return x

class OptimizedGPT_TE_PyResMoE(nn.Module):
    """GPT with Full Attention and Pyramid-Residual MoE MLPs (FP8-ready)."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings (BF16), tie weights
        wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = wte.weight
        
        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(),
            ln_f = te.RMSNorm(config.n_embd, eps=1e-6),
        ))
        
        # RoPE cache
        self.register_buffer(
            "rope_cache",
            RoPE.create_cos_sin_cache(config.block_size, config.head_dim, base=config.rope_theta, device="cpu"),
            persistent=False,
        )
        
        # Build blocks: choose MoE placement
        moe_layers = set(config.moe_layers) if (config.moe_layers is not None) else set()
        for layer_idx in range(config.n_layer):
            if layer_idx in moe_layers:
                self.transformer.h.append(FullAttnMoEBlock(config))
            else:
                self.transformer.h.append(FullAttnFFNBlock(config))
        
        self.apply(self._init_weights)
        
        if config.fuse_wgrad_accumulation:
            print("Initializing main_grad tensors for gradient accumulation fusion...")
            for p in self.parameters():
                if p.requires_grad:
                    p.main_grad = torch.zeros_like(p, dtype=torch.float32, device=p.device)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block size"
        
        # X: [B,T,C]
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        
        # Precompute/transfer RoPE cache
        rope_cache = self.rope_cache
        if rope_cache.device != x.device or rope_cache.size(0) < T:
            self.rope_cache = RoPE.create_cos_sin_cache(self.config.block_size, self.config.head_dim, base=self.config.rope_theta, device=x.device)
            rope_cache = self.rope_cache
        
        for block in self.transformer.h:
            x = block(x, rope_cache)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
            return idx
        finally:
            if was_training:
                self.train()
