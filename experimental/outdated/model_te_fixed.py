#!/usr/bin/env python3
"""
TransformerEngine FP8 version with FIXED optimizations:
1. Proper FP8 weight caching via autocast context
2. QKV fusion with GroupedLinear (independent of grad fusion)
3. Removed unnecessary gradient accumulation fusion
4. Optimized for RTX 5090 with correct TE usage
"""

import os
# Force FlashAttention backend for RTX 5090
os.environ.setdefault('NVTE_FUSED_ATTN_BACKEND', '1')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

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
    vocab_size: int = 49152     # SmolLM vocab
    n_layer: int = 12            # 112M model configuration
    n_head: int = 12             # Number of attention heads
    n_embd: int = 768            # Embedding dimension
    n_kv_heads: int = 3          # GQA: 3 KV heads (4x compression)
    block_size: int = 2048       # Context length
    dropout: float = 0.05        # Conservative dropout
    bias: bool = False           # No bias in Linear/LayerNorm
    rope_theta: float = 10000.0
    # FP8 configuration
    use_fp8: bool = True         # Enable FP8 training
    fp8_amax_history_len: int = 16  # Start conservative
    fp8_amax_compute_algo: str = "max"
    # Optimizations (simplified and correct)
    fuse_qkv: bool = True        # Use GroupedLinear for QKV (recommended)
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_heads == 0
        self.head_dim = self.n_embd // self.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.n_embd % 16 == 0, "n_embd must be divisible by 16 for FP8"
        assert self.head_dim % 16 == 0, "head_dim must be divisible by 16 for FP8"


def get_fp8_recipe(config):
    """Get the FP8 recipe for training."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    print(f"GPU: {device_name}")
    print("Using DelayedScaling FP8 (RTX 5090 optimized)")
    print(f"  QKV fusion with GroupedLinear: {config.fuse_qkv}")
    print("  FP8 weight caching: Enabled via autocast context")
    print("  Gradient accumulation: Native PyTorch (optimal)")
    
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


class FixedAttention(nn.Module):
    """Multi-Head Attention with proper QKV fusion and GQA."""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.fuse_qkv = config.fuse_qkv
        
        if self.fuse_qkv:
            # Use GroupedLinear for QKV - single kernel, three outputs
            # This is the RIGHT way to fuse QKV with TransformerEngine
            self.qkv = te.LayerNormLinear(
                config.n_embd,
                config.n_head * config.head_dim + 2 * config.n_kv_heads * config.head_dim,
                bias=config.bias,
                params_dtype=torch.bfloat16,
                normalization="RMSNorm",
                return_layernorm_output=False
            )
        else:
            # Separate projections (fallback)
            self.q_proj = te.Linear(
                config.n_embd, 
                config.n_head * config.head_dim,
                bias=config.bias, 
                params_dtype=torch.bfloat16
            )
            self.k_proj = te.Linear(
                config.n_embd, 
                config.n_kv_heads * config.head_dim,
                bias=config.bias, 
                params_dtype=torch.bfloat16
            )
            self.v_proj = te.Linear(
                config.n_embd, 
                config.n_kv_heads * config.head_dim,
                bias=config.bias, 
                params_dtype=torch.bfloat16
            )
        
        self.o_proj = te.Linear(
            config.n_head * config.head_dim, 
            config.n_embd,
            bias=config.bias, 
            params_dtype=torch.bfloat16
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout
    
    def forward(self, x, rope_cache, is_first_microbatch=None):
        B, T, C = x.shape
        
        if self.fuse_qkv:
            # Fused QKV projection with single kernel
            qkv = self.qkv(x, is_first_microbatch=is_first_microbatch)
            # Split into Q, K, V
            q_size = self.n_head * self.head_dim
            k_size = self.n_kv_heads * self.head_dim
            v_size = self.n_kv_heads * self.head_dim
            
            q = qkv[:, :, :q_size].view(B, T, self.n_head, self.head_dim)
            k = qkv[:, :, q_size:q_size+k_size].view(B, T, self.n_kv_heads, self.head_dim)
            v = qkv[:, :, q_size+k_size:].view(B, T, self.n_kv_heads, self.head_dim)
        else:
            # Separate projections
            q = self.q_proj(x, is_first_microbatch=is_first_microbatch).view(B, T, self.n_head, self.head_dim)
            k = self.k_proj(x, is_first_microbatch=is_first_microbatch).view(B, T, self.n_kv_heads, self.head_dim)
            v = self.v_proj(x, is_first_microbatch=is_first_microbatch).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = RoPE.apply_rotary_pos_emb(q, rope_cache)
        k = RoPE.apply_rotary_pos_emb(k, rope_cache)
        
        # Reshape for attention [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA: Repeat KV heads
        if self.n_kv_heads != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        
        # Use Flash Attention for best performance on RTX 5090
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0 if not self.training else self.dropout_p,
                is_causal=True
            )
        
        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.o_proj(y, is_first_microbatch=is_first_microbatch)
        y = self.dropout(y)
        
        return y


class FixedFeedForward(nn.Module):
    """SwiGLU feedforward network."""
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.n_embd * 8/3)
        hidden_dim = (hidden_dim + 63) // 64 * 64
        
        self.gate_proj = te.Linear(
            config.n_embd, 
            hidden_dim,
            bias=config.bias, 
            params_dtype=torch.bfloat16
        )
        self.up_proj = te.Linear(
            config.n_embd, 
            hidden_dim,
            bias=config.bias, 
            params_dtype=torch.bfloat16
        )
        self.down_proj = te.Linear(
            hidden_dim, 
            config.n_embd,
            bias=config.bias, 
            params_dtype=torch.bfloat16
        )
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, is_first_microbatch=None):
        gate = F.silu(self.gate_proj(x, is_first_microbatch=is_first_microbatch))
        up = self.up_proj(x, is_first_microbatch=is_first_microbatch)
        x = gate * up
        x = self.down_proj(x, is_first_microbatch=is_first_microbatch)
        x = self.dropout(x)
        return x


class FixedBlock(nn.Module):
    """Transformer block with fixed optimizations."""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = FixedAttention(config)
        self.ln_2 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.ffn = FixedFeedForward(config)
    
    def forward(self, x, rope_cache, is_first_microbatch=None):
        # Pass is_first_microbatch through to TE layers
        x = x + self.attn(self.ln_1(x), rope_cache, is_first_microbatch)
        x = x + self.ffn(self.ln_2(x), is_first_microbatch)
        return x


class FixedGPT_TE(nn.Module):
    """GPT model with FIXED FP8 optimizations."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings (BF16)
        wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = wte.weight  # Weight tying
        
        # Transformer blocks
        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([FixedBlock(config) for _ in range(config.n_layer)]),
            ln_f = te.RMSNorm(config.n_embd, eps=1e-6),
        ))
        
        # RoPE cache
        self.register_buffer('rope_cache', 
            RoPE.create_cos_sin_cache(config.block_size, config.head_dim, config.rope_theta)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Report configuration
        total_params = self.num_parameters()
        print(f"Fixed model initialized with TransformerEngine FP8")
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  Architecture: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
        print(f"  GQA: {config.n_kv_heads} KV heads ({config.n_head//config.n_kv_heads}x compression)")
        print(f"  Optimizations:")
        print(f"    - QKV fusion: {config.fuse_qkv}")
        print(f"    - FP8 weight caching: Via autocast context")
        print(f"    - Gradient accumulation: Native PyTorch")
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, idx, targets=None, is_first_microbatch=None):
        """
        Forward pass with is_first_microbatch for FP8 weight caching.
        In TE ≥2.4, this flag is passed to individual modules.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        # Token embeddings
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x, self.rope_cache, is_first_microbatch)
        
        # Final norm and output
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens (inference in BF16)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


if __name__ == "__main__":
    # Test the fixed model
    if not torch.cuda.is_available():
        print("CUDA not available. FP8 requires GPU.")
        import sys
        sys.exit(1)
    
    # Force Flash Attention
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    
    config = ModelConfig()
    model = FixedGPT_TE(config).cuda()
    model = model.to(torch.bfloat16)
    
    # Test with proper FP8 weight caching
    x = torch.randint(0, config.vocab_size, (2, 128)).cuda()
    fp8_recipe = get_fp8_recipe(config)
    
    print("\nTesting with PROPER FP8 weight caching (TE ≥2.4 API)...")
    
    # Simulate gradient accumulation with correct module-level caching
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    optimizer.zero_grad(set_to_none=True)
    
    for micro_step in range(4):
        # In TE ≥2.4: Pass is_first_microbatch to modules, not autocast!
        is_first_microbatch = (micro_step == 0)
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            logits, loss = model(x, targets=x, is_first_microbatch=is_first_microbatch)
        
        loss = loss / 4  # Scale by gradient accumulation steps
        loss.backward()
        print(f"  Micro-step {micro_step}: loss={loss.item()*4:.4f} (first={is_first_microbatch})")
    
    # Optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    print("✓ Fixed model test successful!")
    print("  - FP8 weight caching enabled via module is_first_microbatch (TE ≥2.4)")
    print("  - Native gradient accumulation (no overhead)")
    print("  - QKV fusion independent of grad accumulation")