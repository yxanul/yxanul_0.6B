#!/usr/bin/env python3
"""
Native TransformerEngine model for FP8 acceleration.
Uses TE layers throughout for maximum performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    HAS_TE = True
except ImportError:
    HAS_TE = False
    print("Warning: TransformerEngine not found. Install with: pip install transformer-engine")

@dataclass
class TEModelConfig:
    vocab_size: int = 50264  # Padded for FP8 alignment (divisible by 8)
    n_layer: int = 12        # GPT-2 small
    n_head: int = 12         # Number of attention heads  
    n_embd: int = 768        # Embedding dimension (divisible by 16 for FP8)
    n_kv_heads: int = 12     # For now, no GQA in TE (use same as n_head)
    block_size: int = 128    # Context length
    dropout: float = 0.0     # Dropout rate
    bias: bool = False       # No bias in layers
    rope_theta: float = 10000.0
    hidden_size: int = None  # Will be set to n_embd
    ffn_hidden_size: int = None  # Will be calculated
    
    def __post_init__(self):
        self.hidden_size = self.n_embd
        # SwiGLU expansion with FP8 alignment
        self.ffn_hidden_size = int(self.n_embd * 8/3)
        # Round to multiple of 64 for tensor core efficiency
        self.ffn_hidden_size = (self.ffn_hidden_size + 63) // 64 * 64
        # Ensure FP8 alignment (divisible by 16)
        assert self.ffn_hidden_size % 16 == 0
        
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"


class RoPE:
    """Rotary Position Embeddings for TE model."""
    @staticmethod
    def create_cos_sin_cache(seq_len, n_elem, base=10000, device='cpu', dtype=torch.float32):
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device, dtype=dtype) / n_elem))
        seq_idx = torch.arange(seq_len, device=device, dtype=dtype)
        idx_theta = torch.outer(seq_idx, theta)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        return cache
    
    @staticmethod  
    def apply_rotary_pos_emb(x, cos_sin_cache):
        batch, seq_len, n_heads, head_dim = x.shape
        x = x.reshape(batch, seq_len, n_heads, head_dim // 2, 2)
        
        cos_cache = cos_sin_cache[:seq_len, :, 0].to(x.device, x.dtype)
        sin_cache = cos_sin_cache[:seq_len, :, 1].to(x.device, x.dtype)
        
        x_rot = torch.stack([
            x[..., 0] * cos_cache.unsqueeze(0).unsqueeze(2) - x[..., 1] * sin_cache.unsqueeze(0).unsqueeze(2),
            x[..., 0] * sin_cache.unsqueeze(0).unsqueeze(2) + x[..., 1] * cos_cache.unsqueeze(0).unsqueeze(2)
        ], dim=-1)
        
        return x_rot.reshape(batch, seq_len, n_heads, head_dim)


class TEAttention(nn.Module):
    """TransformerEngine attention layer with FP8 support."""
    
    def __init__(self, config: TEModelConfig):
        super().__init__()
        assert HAS_TE, "TransformerEngine required for TEAttention"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        
        # Use TE Linear layers for FP8 acceleration
        self.qkv_proj = te.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        
        self.o_proj = te.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(config.head_dim)
        
    def forward(self, x, rope_cache):
        B, T, C = x.shape
        # Ensure contiguous and 2D for TE GEMM (M=B*T, K=C)
        x2d = x.contiguous().view(B * T, C)
        
        # Compute Q, K, V in one projection
        qkv = self.qkv_proj(x2d)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, B, T, n_head, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        q = RoPE.apply_rotary_pos_emb(q, rope_cache)
        k = RoPE.apply_rotary_pos_emb(k, rope_cache)
        
        # Reshape for attention
        q = q.transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Flash Attention (if available) or standard attention
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Fallback to manual attention
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(
                torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(),
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y2d = y.contiguous().view(B * T, C)
        y2d = self.o_proj(y2d)
        y = y2d.view(B, T, C)
        y = self.resid_dropout(y)
        
        return y


class TEMLP(nn.Module):
    """TransformerEngine MLP with SwiGLU activation."""
    
    def __init__(self, config: TEModelConfig):
        super().__init__()
        assert HAS_TE, "TransformerEngine required for TEMLP"
        
        # SwiGLU uses 2 projections for gating
        self.gate_up_proj = te.Linear(
            config.n_embd,
            2 * config.ffn_hidden_size,
            bias=config.bias,
        )
        
        self.down_proj = te.Linear(
            config.ffn_hidden_size,
            config.n_embd,
            bias=config.bias,
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        x2d = x.contiguous().view(B * T, C)
        # Combined gate and up projection
        gate_up = self.gate_up_proj(x2d)
        
        # Split into gate and up
        gate, up = gate_up.chunk(2, dim=-1)
        
        # SwiGLU activation
        x_act = F.silu(gate) * up
        
        # Down projection
        x2d = self.down_proj(x_act)
        x2d = self.dropout(x2d)
        return x2d.view(B, T, C)


class TEBlock(nn.Module):
    """TransformerEngine transformer block."""
    
    def __init__(self, config: TEModelConfig):
        super().__init__()
        assert HAS_TE, "TransformerEngine required for TEBlock"
        
        # Use TE LayerNorm for FP8 support
        self.ln_1 = te.LayerNorm(
            config.n_embd,
            eps=1e-6,
            zero_centered_gamma=True  # Better for FP8
        )
        
        self.attn = TEAttention(config)
        
        self.ln_2 = te.LayerNorm(
            config.n_embd,
            eps=1e-6,
            zero_centered_gamma=True
        )
        
        self.mlp = TEMLP(config)
        
    def forward(self, x, rope_cache):
        # Pre-norm architecture
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class TETransformerGPT(nn.Module):
    """TransformerEngine GPT model optimized for FP8."""
    
    def __init__(self, config: TEModelConfig):
        super().__init__()
        assert HAS_TE, "TransformerEngine required for TETransformerGPT"
        
        self.config = config
        
        # Token embeddings (regular PyTorch, as TE doesn't have embedding layer)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TEBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = te.LayerNorm(
            config.n_embd,
            eps=1e-6,
            zero_centered_gamma=True
        )
        
        # Output head - use regular Linear so we can tie weights
        # TE Linear doesn't support weight tying easily
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )
        
        # Weight tying to save parameters (like original GPT-2)
        self.lm_head.weight = self.wte.weight
        
        # RoPE cache
        self.register_buffer('rope_cache',
            RoPE.create_cos_sin_cache(
                config.block_size,
                config.head_dim,
                config.rope_theta
            )
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections
        for name, param in self.named_parameters():
            if 'o_proj.weight' in name or 'down_proj.weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print(f"TE Model initialized: {self.num_parameters():,} parameters")
        print(f"Using TransformerEngine with FP8 support")
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            if hasattr(module, 'weight'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, te.LayerNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        # Token embeddings
        tok_emb = self.wte(idx)
        x = self.drop(tok_emb)
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, self.rope_cache)
        
        # Final norm and output
        x = self.ln_f(x)
        # Flatten to 2D for TE GEMM
        B, T, C = x.shape
        x2d = x.contiguous().view(B * T, C)
        logits2d = self.lm_head(x2d)
        logits = logits2d.view(B, T, self.config.vocab_size)
        
        # Calculate loss if targets provided
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
        """Generate tokens."""
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop probabilities to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def create_te_model(vocab_size=50264):
    """Create a TransformerEngine model."""
    if not HAS_TE:
        raise ImportError("TransformerEngine not available. Install with: pip install transformer-engine")
    
    config = TEModelConfig(vocab_size=vocab_size)
    model = TETransformerGPT(config)
    return model, config


if __name__ == "__main__":
    # Test the model
    if HAS_TE:
        model, config = create_te_model()
        
        # Test forward pass
        x = torch.randint(0, config.vocab_size, (2, 100)).cuda()
        
        # Test with FP8 autocast
        fp8_recipe = recipe.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=recipe.Format.E4M3,
        )
        
        with te.fp8_autocast(enabled=True, calibrating=True, fp8_recipe=fp8_recipe):
            logits, loss = model(x, targets=x)
        
        print(f"Logits shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")
        
        # Test generation
        context = torch.zeros((1, 1), dtype=torch.long).cuda()
        generated = model.generate(context, max_new_tokens=20)
        print(f"Generated shape: {generated.shape}")
    else:
        print("TransformerEngine not available. Cannot test TE model.")
