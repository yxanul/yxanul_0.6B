#!/usr/bin/env python3
"""
CPU-compatible inference model that can load weights from TransformerEngine training.
This replaces TE layers with standard PyTorch layers for CPU inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 49152     # SmolLM vocab
    n_layer: int = 12            # 112M model configuration
    n_head: int = 12             # Number of attention heads
    n_embd: int = 768            # Embedding dimension
    n_kv_heads: int = 3          # GQA: 3 KV heads (4x compression)
    block_size: int = 2048       # Context length
    dropout: float = 0.0         # No dropout for inference
    bias: bool = False           # No bias in Linear/LayerNorm
    rope_theta: float = 10000.0
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_heads == 0
        self.head_dim = self.n_embd // self.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


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


class Attention(nn.Module):
    """Multi-Head Attention with GQA - CPU compatible."""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        
        # Standard PyTorch Linear layers
        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout
    
    def forward(self, x, rope_cache):
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        
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
        
        # Scaled dot-product attention
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


class FeedForward(nn.Module):
    """SwiGLU feedforward - CPU compatible."""
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.n_embd * 8/3)
        hidden_dim = (hidden_dim + 63) // 64 * 64
        
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block - CPU compatible."""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=1e-6)
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=1e-6)
        self.ffn = FeedForward(config)
    
    def forward(self, x, rope_cache):
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = x + self.ffn(self.ln_2(x))
        return x


class GPTInference(nn.Module):
    """GPT model for CPU inference - compatible with TE trained weights."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = wte.weight  # Weight tying
        
        # Transformer blocks
        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, eps=1e-6),
        ))
        
        # RoPE cache
        self.register_buffer('rope_cache', 
            RoPE.create_cos_sin_cache(config.block_size, config.head_dim, config.rope_theta)
        )
    
    def load_from_te_checkpoint(self, checkpoint_path):
        """Load weights from TransformerEngine checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']
        
        # Create mapping from TE names to our names
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Skip TE-specific parameters
            if any(skip in key.lower() for skip in ['fp8', 'scale', '_extra_state']):
                continue
                
            # Map the weights - TE uses same structure
            new_state_dict[key] = value
        
        # Load the mapped weights
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Iterations: {checkpoint.get('iter_num', 'unknown')}")
        print(f"  Val loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
        
        if missing:
            print(f"  Warning: Missing keys: {len(missing)} keys")
            if len(missing) < 10:  # Only show if not too many
                for key in missing:
                    print(f"    - {key}")
        
        if unexpected:
            print(f"  Info: Skipped TE-specific keys: {len(unexpected)} keys")
        
        return checkpoint
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        # Token embeddings
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x, self.rope_cache)
        
        # Final norm and output
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
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
    # Test the inference model
    config = ModelConfig()
    model = GPTInference(config)
    
    print(f"Inference model initialized")
    print(f"  Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"  Architecture: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
    print(f"  This model can run on CPU!")
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (1, 128))
    logits, loss = model(x, targets=x)
    print(f"✓ CPU inference test successful!")
    print(f"  Output shape: {logits.shape}")