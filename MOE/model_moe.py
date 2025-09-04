"""GLM-mini-ish: Decoder-only Transformer with
- Dense stem (first block)
- PR-MoE (Pyramid-Residual MoE) in the remaining blocks
- Sigmoid router + loss-free load balancing (bias nudging)
- Grouped-Query Attention (GQA)
- Partial RoPE (apply rotary only to a fraction of heads)
- QK-Norm on attention logits
- One MTP MoE layer on top for multi-token prediction (GLM-4.5 style)
- FP8-friendly (Transformer Engine Linear layers, BF16 params)
This is a full rewrite derived from the user's initial model file."
"""

import math
import copy
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Transformer Engine (TE) is required for FP8-friendly Linear layers.
# -----------------------------------------------------------------------------
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
except Exception as e:
    raise ImportError("TransformerEngine is required (pip install transformer-engine).") from e


# -----------------------------------------------------------------------------
# FP8 recipe helper (you'll use this in your training loop, not inside the model)
# -----------------------------------------------------------------------------
def get_fp8_recipe(amax_history_len: int = 16, amax_compute_algo: str = "max"):
    """
    Create a DelayedScaling FP8 recipe. Use this via:
      with te.fp8_autocast(enabled=True, fp8_recipe=get_fp8_recipe(...)):
          loss = model(...)
    """
    return DelayedScaling(
        fp8_format=Format.HYBRID,   # E4M3 forward, E5M2 backward
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
    )


# -----------------------------------------------------------------------------
# Configuration (GLM-mini-ish defaults stay < ~400M total params)
# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Core model sizes (GLM-mini-ish, depth-heavy, many small heads: head_dim=32)
    vocab_size: int = 32000              # Byte-level BPE tokenizer
    n_layer: int = 24
    n_embd: int = 896
    n_head: int = 28
    n_kv_heads: int = 7                # GQA: KV heads << Q heads (28/7 = 4x repeat)
    block_size: int = 2048              # Stage 1 pretraining: shorter sequences
    dropout: float = 0.05
    bias: bool = False

    # RoPE and attention stabilizers
    rope_theta: float = 10000.0
    partial_rope_ratio: float = 0.75    # apply RoPE to first 75% of heads
    use_qk_norm: bool = True            # normalize Q and K (RMS-type)

    # FP8-friendly invariants
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"

    # TE / optimizer-side accel toggles
    fuse_wgrad_accumulation: bool = True
    cache_fp8_weights: bool = True

    # Dense stem (first N blocks), MoE elsewhere
    dense_stem_layers: int = 1          # GLM: 1 dense + rest MoE
    moe_layers: Optional[List[int]] = None  # None => MoE in all non-stem layers

    # PR-MoE sizing (base + experts per MoE block)
    base_expansion: float = 1.25         # always-on base MLP width multiplier
    moe_min_expansion: float = 1.25      # expert widths: geometric [min..max]
    moe_max_expansion: float = 1.75
    moe_num_experts: int = 3

    # Sigmoid router (variable compute) + loss-free balancing
    router_type: str = "sigmoid"         # {"sigmoid","softmax_topk"}
    router_tau: float = 1.0
    router_threshold: float = 0.50       # raise to reduce activations
    router_stochastic: bool = True       # Bernoulli during training
    router_max_active_experts: int = 1   # hard cap per token (GLM-ish efficient)
    balance_loss_free: bool = True       # aux-free uniformization via bias
    balance_ema_beta: float = 0.9
    balance_alpha: float = 0.1

    # One MTP MoE layer at the top (GLM-4.5 style)
    use_mtp: bool = True
    mtp_offsets: Tuple[int, ...] = (2, 3, 4)
    mtp_weights: Tuple[float, ...] = (1.0, 0.5, 0.25)
    mtp_layer_expansion_base: float = 1.0
    mtp_layer_num_experts: int = 2

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        self.head_dim: int = self.n_embd // self.n_head
        # FP8 kernels love multiples of 16
        assert self.n_embd % 16 == 0, "n_embd must be divisible by 16 for FP8 kernels"
        assert self.head_dim % 16 == 0, "head_dim must be divisible by 16 for FP8 kernels"
        # RoPE expects even head_dim for interleaving
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        # MTP sanity
        if self.use_mtp:
            assert len(self.mtp_offsets) == len(self.mtp_weights), "MTP offsets/weights mismatch"


# -----------------------------------------------------------------------------
# Rotary Position Embeddings (with helpers for partial-head application)
# -----------------------------------------------------------------------------
class RoPE:
    @staticmethod
    def create_cos_sin_cache(seq_len: int, n_elem: int, base: float = 10000.0, device: str = "cpu"):
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
        seq_idx = torch.arange(seq_len, device=device).float()
        idx_theta = torch.outer(seq_idx, theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)  # [T, n_elem/2, 2]
        return cache

    @staticmethod
    def apply_rotary_pos_emb(x_bh_t_d: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to tensor shaped [B, H, T, D].
        """
        B, H, T, D = x_bh_t_d.shape
        x = x_bh_t_d.transpose(2, 3).contiguous().view(B, H, D // 2, 2, T)
        cos = cos_sin_cache[:T, :, 0].transpose(0, 1)  # [D/2, T]
        sin = cos_sin_cache[:T, :, 1].transpose(0, 1)  # [D/2, T]
        xr, xi = x[..., 0, :], x[..., 1, :]
        # rotate
        xo_r = xr * cos - xi * sin
        xo_i = xr * sin + xi * cos
        out = torch.stack([xo_r, xo_i], dim=3).reshape(B, H, D, T).transpose(2, 3).contiguous()
        return out


# -----------------------------------------------------------------------------
# QK-Norm (simple RMS-style normalization of Q and K)
# -----------------------------------------------------------------------------
class QKNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q,k: [B, H, T, D]
        q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return q * self.scale, k * self.scale


# -----------------------------------------------------------------------------
# Attention with GQA, Partial RoPE, and QK-Norm (FP8-friendly via TE Linear)
# -----------------------------------------------------------------------------
class OptimizedAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.dropout_p = config.dropout

        # fused QKV (BF16 params; FP8 handled by autocast recipe in training loop)
        self.qkv_proj = te.Linear(
            in_features=self.n_embd,
            out_features=(self.n_head + 2 * self.n_kv_heads) * self.head_dim,
            bias=config.bias,
            params_dtype=torch.bfloat16,
        )
        self.o_proj = te.Linear(
            in_features=self.n_head * self.head_dim,
            out_features=self.n_embd,
            bias=config.bias,
            params_dtype=torch.bfloat16,
        )
        self.drop = nn.Dropout(config.dropout)

        # stabilizers
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)
        self.partial_rope_ratio = float(config.partial_rope_ratio)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # [B, T, (H + 2*Hkv)*D]
        q, k, v = torch.split(qkv, [self.n_head * self.head_dim,
                                    self.n_kv_heads * self.head_dim,
                                    self.n_kv_heads * self.head_dim], dim=-1)

        # [B, H, T, D] / [B, Hkv, T, D]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Partial RoPE on first fraction of heads
        H_rope_q = int(round(self.n_head * self.partial_rope_ratio))
        H_rope_k = int(round(self.n_kv_heads * self.partial_rope_ratio))
        if H_rope_q > 0:
            q[:, :H_rope_q] = RoPE.apply_rotary_pos_emb(q[:, :H_rope_q], rope_cache)
        if H_rope_k > 0:
            k[:, :H_rope_k] = RoPE.apply_rotary_pos_emb(k[:, :H_rope_k], rope_cache)

        # QK-Norm (stabilize attention logits range)
        if self.use_qk_norm:
            q, k = self.qk_norm(q, k)

        # GQA: repeat KV heads to match Q heads
        if self.n_kv_heads != self.n_head:
            repeat = self.n_head // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Flash/SDP attention (causal)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True
        )  # [B, H, T, D]

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return self.drop(y)


# -----------------------------------------------------------------------------
# Dense FFN (SwiGLU)
# -----------------------------------------------------------------------------
class DenseFFN(nn.Module):
    def __init__(self, config: ModelConfig, expansion: float = 8.0/3.0):
        super().__init__()
        C = config.n_embd
        H = int(round(C * expansion / 64) * 64)  # align to 64 for TE kernels
        self.gate = te.Linear(C, H, bias=config.bias, params_dtype=torch.bfloat16)
        self.up   = te.Linear(C, H, bias=config.bias, params_dtype=torch.bfloat16)
        self.down = te.Linear(H, C, bias=config.bias, params_dtype=torch.bfloat16)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        u = self.up(x)
        h = F.silu(g) * u
        h = self.down(h)
        return x + self.drop(h)


# -----------------------------------------------------------------------------
# PR-MoE: Base (always-on) dense MLP + sparse experts (SwiGLU) with sigmoid router
# -----------------------------------------------------------------------------
class PyramidResidualMoE(nn.Module):
    """
    PR-MoE with:
      - Always-on base MLP: width = base_expansion * d_model
      - E experts with widths geometrically spaced in [min_expansion .. max_expansion] * d_model
      - Sigmoid router with independent activations per expert (variable compute)
      - Loss-free balancing: per-expert bias nudged by EMA load (no aux loss)
      - Hard cap on max active experts per token (compute envelope)
      Output: x + base(x) + sum_e w_e * expert_e(x)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        C = config.n_embd
        E = config.moe_num_experts

        # Base (always-on) SwiGLU MLP
        base_h = int(round(C * config.base_expansion / 64) * 64)
        self.base_gate = te.Linear(C, base_h, bias=config.bias, params_dtype=torch.bfloat16)
        self.base_up   = te.Linear(C, base_h, bias=config.bias, params_dtype=torch.bfloat16)
        self.base_down = te.Linear(base_h, C, bias=config.bias, params_dtype=torch.bfloat16)

        # Router head
        # Use regular Linear for router since E=3 doesn't meet FP8 requirements (need % 16)
        self.router = nn.Linear(C, E, bias=config.bias)
        self.register_buffer("balance_bias", torch.zeros(E), persistent=False)
        self.register_buffer("ema_load", torch.full((E,), 1.0/E), persistent=False)

        # Experts (geometric widths)
        if E == 1:
            ratios = [config.moe_min_expansion]
        else:
            ratios = torch.logspace(
                math.log10(config.moe_min_expansion),
                math.log10(max(config.moe_min_expansion, config.moe_max_expansion)),
                steps=E
            ).tolist()
        self.experts = nn.ModuleList()
        for r in ratios:
            h = int(round(C * float(r) / 64) * 64)
            # Use regular nn.Linear for experts to avoid cuBLAS issues with variable batch sizes
            self.experts.append(nn.ModuleDict(dict(
                gate = nn.Linear(C, h, bias=config.bias),
                up   = nn.Linear(C, h, bias=config.bias),
                down = nn.Linear(h, C, bias=config.bias),
            )))

    # ----- helpers -----
    def _base(self, x: torch.Tensor) -> torch.Tensor:
        g = self.base_gate(x); u = self.base_up(x)
        return self.base_down(F.silu(g) * u)

    @torch.no_grad()
    def _update_balance(self, active: torch.Tensor):
        # active: [B,T,E] 0/1
        loads = active.sum(dim=(0,1))  # [E]
        total = loads.sum().clamp_min(1e-6)
        loads = loads / total
        beta = self.cfg.balance_ema_beta
        self.ema_load.mul_(beta).add_((1 - beta) * loads)
        target = 1.0 / active.size(-1)
        self.balance_bias.add_(-self.cfg.balance_alpha * (self.ema_load - target))

    def _route_sigmoid(self, x: torch.Tensor):
        # logits + bias → sigmoid probs → active mask (+ ensure ≥1) → cap → renormalize
        logits = (self.router(x) + self.balance_bias.view(1,1,-1)) / max(self.cfg.router_tau, 1e-6)
        probs = torch.sigmoid(logits)

        if self.cfg.router_stochastic and self.training:
            active = torch.bernoulli(probs.clamp(0,1).float()).to(probs.dtype)
        else:
            active = (probs >= self.cfg.router_threshold).to(probs.dtype)

        # ensure at least one expert per token
        none = (active.sum(dim=-1, keepdim=True) == 0)
        if none.any():
            top1 = torch.argmax(probs, dim=-1, keepdim=True)
            fix = F.one_hot(top1.squeeze(-1), probs.size(-1)).to(probs.dtype).unsqueeze(2).squeeze(2)
            active = torch.where(none, fix, active)

        # hard cap k active experts/token
        if self.cfg.router_max_active_experts and self.cfg.router_max_active_experts > 0:
            k = min(self.cfg.router_max_active_experts, probs.size(-1))
            topv, topi = torch.topk(probs, k=k, dim=-1)
            cap = torch.zeros_like(active).scatter_(-1, topi, 1.0)
            active = active * cap

        weights = probs * active
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.cfg.balance_loss_free and self.training:
            self._update_balance(active)

        return weights, active

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        base_out = self._base(x)

        if self.cfg.router_type == "sigmoid":
            weights, active = self._route_sigmoid(x)
        else:
            # fallback: softmax top-k (not GLM-style; kept for completeness)
            gate = F.softmax(self.router(x), dim=-1)  # [B,T,E]
            k = 1
            topv, topi = torch.topk(gate, k=k, dim=-1)
            weights = torch.zeros_like(gate).scatter_(-1, topi, topv)
            active = (weights > 0).to(weights.dtype)

        # sparse expert compute
        x_flat = x.view(B*T, C)
        out = torch.zeros_like(x_flat)
        w_flat = weights.view(B*T, -1)
        a_flat = active.view(B*T, -1)

        for e, expert in enumerate(self.experts):
            sel = a_flat[:, e] > 0
            if not sel.any():
                continue
            idx = sel.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            
            w = w_flat[idx, e].unsqueeze(1)
            tokens = x_flat.index_select(0, idx)
            
            # Pad tokens for FP8/cuBLAS alignment
            # Need first dim % 8 == 0 for FP8, but also % 16 for some cuBLAS kernels
            n_tokens = tokens.shape[0]
            # Align to 16 for maximum compatibility
            alignment = 16
            remainder = n_tokens % alignment
            if remainder != 0:
                pad_size = alignment - remainder
                padding = torch.zeros(pad_size, tokens.shape[1], 
                                     dtype=tokens.dtype, device=tokens.device)
                tokens_padded = torch.cat([tokens, padding], dim=0)
            else:
                tokens_padded = tokens
            
            # Process with padding
            h = F.silu(expert["gate"](tokens_padded)) * expert["up"](tokens_padded)
            h = expert["down"](h)
            
            # Remove padding if added
            if remainder != 0:
                h = h[:n_tokens]
            
            out.index_add_(0, idx, w * h)

        return x + base_out + out.view(B, T, C)


# -----------------------------------------------------------------------------
# Transformer blocks: Dense stem vs PR-MoE blocks
# -----------------------------------------------------------------------------
class FullAttnFFNBlock(nn.Module):
    """Dense block: GQA attention + Dense FFN."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = OptimizedAttention(config)
        self.ln_2 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.ffn  = DenseFFN(config, expansion=8.0/3.0)  # classic 4x with SwiGLU ~ 8/3

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor):
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = self.ffn(self.ln_2(x))
        return x


class FullAttnMoEBlock(nn.Module):
    """MoE block: GQA attention + PR-MoE MLP."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = OptimizedAttention(config)
        self.ln_2 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.moe  = PyramidResidualMoE(config)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor):
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = self.moe(self.ln_2(x))  # residuals inside
        return x


# -----------------------------------------------------------------------------
# One MTP MoE layer (GLM-4.5: add MoE layer as MTP layer)
# -----------------------------------------------------------------------------
class MTPMoELayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        cfg = copy.deepcopy(config)
        cfg.base_expansion = config.mtp_layer_expansion_base
        cfg.moe_num_experts = config.mtp_layer_num_experts
        cfg.moe_min_expansion = 1.0
        cfg.moe_max_expansion = 1.25
        self.norm = te.RMSNorm(cfg.n_embd, eps=1e-6)
        self.moe  = PyramidResidualMoE(cfg)
        # per-offset bias (tiny params) when using tied lm_head
        self.offset_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(cfg.vocab_size)) for _ in cfg.mtp_offsets
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(self.norm(x))


# -----------------------------------------------------------------------------
# Top-level model
# -----------------------------------------------------------------------------
class OptimizedGPT_GLMini_PRMoE(nn.Module):
    """
    Decoder-only Transformer:
      - First N=dense_stem_layers blocks: Dense FFN
      - Remaining blocks: PR-MoE with sigmoid routing
      - One top MoE layer for MTP (multi-token prediction) training
      - Tied embeddings & output head
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings + tied LM head
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # Transformer stack
        blocks = nn.ModuleList()
        for i in range(config.n_layer):
            if i < config.dense_stem_layers:
                blocks.append(FullAttnFFNBlock(config))
            else:
                blocks.append(FullAttnMoEBlock(config))

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = te.RMSNorm(config.n_embd, eps=1e-6),
        ))

        # RoPE cache (created lazily on first forward to the device/length needed)
        self.register_buffer(
            "rope_cache",
            RoPE.create_cos_sin_cache(config.block_size, config.head_dim, base=config.rope_theta, device="cpu"),
            persistent=False,
        )

        # One MoE MTP layer (GLM-style)
        self.mtp_layer = MTPMoELayer(config) if config.use_mtp else None

        # Init
        self.apply(self._init_weights)
        if config.fuse_wgrad_accumulation:
            for p in self.parameters():
                if p.requires_grad:
                    p.main_grad = torch.zeros_like(p, dtype=torch.float32, device=p.device)

    # --- inits ---
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # --- utilities ---
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # --- forward ---
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        idx:      [B, T] token ids
        targets:  [B, T] teacher-forced targets (ignore_index=3 for padding)
        returns: (logits, loss)
        """
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block size"

        x = self.wte(idx)                  # [B,T,C]
        x = self.transformer.drop(x)

        # ensure RoPE cache proper device/length (cache is [T, D/2, 2])
        rope_cache = self.rope_cache
        if rope_cache.device != x.device or rope_cache.size(0) < T:
            self.rope_cache = RoPE.create_cos_sin_cache(self.config.block_size, self.config.head_dim,
                                                        base=self.config.rope_theta, device=x.device)
            rope_cache = self.rope_cache

        # blocks
        for block in self.transformer.h:
            x = block(x, rope_cache)

        x = self.transformer.ln_f(x)

        # next-token logits
        logits = self.lm_head(x)           # [B,T,V]

        loss = None
        if targets is not None:
            # standard next-token CE
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=3)  # 3 is pad_token_id

            # One MTP layer on top (GLM-style)
            if self.config.use_mtp and self.mtp_layer is not None:
                mtp_x = self.mtp_layer(x)  # [B,T,C]
                for w, h, bias in zip(self.config.mtp_weights, self.config.mtp_offsets, self.mtp_layer.offset_bias):
                    if h < 2 or h >= T:
                        continue
                    feats = mtp_x[:, :-h, :]
                    logits_h = self.lm_head(feats) + bias.view(1,1,-1)
                    tgt_h = targets[:, h:].contiguous()
                    loss_h = F.cross_entropy(logits_h.float().reshape(-1, logits_h.size(-1)),
                                             tgt_h.view(-1), ignore_index=3)  # 3 is pad_token_id
                    loss = loss + float(w) * loss_h

        return logits, loss

    # --- generation (greedy sampling for quick smoke tests) ---
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None):
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


# -----------------------------------------------------------------------------
# Convenience builder for the GLM-mini-ish config used in our discussion
# -----------------------------------------------------------------------------
def build_glm_mini_prmoe_mtp():
    cfg = ModelConfig(
        vocab_size=32000,
        n_layer=24, n_embd=896, n_head=28, n_kv_heads=7,
        block_size=2048, dropout=0.05, bias=False,
        partial_rope_ratio=0.75, use_qk_norm=True,

        dense_stem_layers=1,
        moe_layers=None,                   # MoE for all non-stem blocks
        moe_num_experts=3,
        base_expansion=1.25,
        moe_min_expansion=1.25,
        moe_max_expansion=1.75,

        router_type="sigmoid",
        router_tau=1.0,
        router_threshold=0.50,
        router_stochastic=True,
        router_max_active_experts=1,
        balance_loss_free=True, balance_ema_beta=0.9, balance_alpha=0.1,

        use_mtp=True,
        mtp_offsets=(2,3,4),
        mtp_weights=(1.0,0.5,0.25),
        mtp_layer_expansion_base=1.0,
        mtp_layer_num_experts=2,
    )
    model = OptimizedGPT_GLMini_PRMoE(cfg)
    return model, cfg


# -----------------------------------------------------------------------------
# Quick self-check (instantiate & count params) if run as script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    model, cfg = build_glm_mini_prmoe_mtp()
    n_params = model.num_parameters()
    print(f"Model parameters (trainable): {n_params/1e6:.2f}M")
    # quick shape smoke test
    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets=idx)  # teacher-forcing sanity check
    print("Logits:", tuple(logits.shape), "Loss:", float(loss))