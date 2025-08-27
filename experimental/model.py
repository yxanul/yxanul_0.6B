#!/usr/bin/env python3
"""
Simple, clean transformer model.
No complications - just what works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257  # GPT-2 vocab
    n_layer: int = 24        # 270M parameters
    n_head: int = 16         # Number of attention heads
    n_embd: int = 1024       # Embedding dimension
    n_kv_heads: int = 4      # GQA: 4 KV heads (4x compression)
    block_size: int = 2048   # Context length
    dropout: float = 0.0     # No dropout for now
    bias: bool = False       # No bias in Linear/LayerNorm
    rope_theta: float = 10000.0
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_heads == 0
        self.head_dim = self.n_embd // self.n_head


class RMSNorm(nn.Module):
    """RMSNorm - simpler and faster than LayerNorm."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RoPE:
    """Rotary Position Embeddings - better than learned positional embeddings."""
    @staticmethod
    def create_cos_sin_cache(seq_len, n_elem, base=10000, device='cpu'):
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
        seq_idx = torch.arange(seq_len, device=device).float()
        idx_theta = torch.outer(seq_idx, theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        return cache
    
    @staticmethod
    def apply_rotary_pos_emb(x, cos_sin_cache):
        # x: [batch, seq_len, n_heads, head_dim]
        batch, seq_len, n_heads, head_dim = x.shape
        x = x.reshape(batch, seq_len, n_heads, head_dim // 2, 2)
        
        cos_cache = cos_sin_cache[:seq_len, :, 0]
        sin_cache = cos_sin_cache[:seq_len, :, 1]
        
        # Apply rotation
        x_rot = torch.stack([
            x[..., 0] * cos_cache.unsqueeze(0).unsqueeze(2) - x[..., 1] * sin_cache.unsqueeze(0).unsqueeze(2),
            x[..., 0] * sin_cache.unsqueeze(0).unsqueeze(2) + x[..., 1] * cos_cache.unsqueeze(0).unsqueeze(2)
        ], dim=-1)
        
        return x_rot.reshape(batch, seq_len, n_heads, head_dim)


class Attention(nn.Module):
    """Multi-Head Attention with Grouped Query Attention (GQA)."""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        
        # GQA: fewer KV heads than Q heads
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout  # Store dropout probability
        self.scale = 1.0 / math.sqrt(config.head_dim)
    
    def forward(self, x, rope_cache):
        B, T, C = x.shape  # batch, sequence, embedding
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        q = RoPE.apply_rotary_pos_emb(q, rope_cache)
        k = RoPE.apply_rotary_pos_emb(k, rope_cache)
        
        # Reshape for attention: [B, n_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA: Repeat KV heads to match Q heads
        if self.n_kv_heads != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        
        # Flash Attention via F.scaled_dot_product_attention
        # This automatically uses Flash Attention 2 when available
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
    """SwiGLU feedforward network - better than standard FFN."""
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.n_embd * 8/3)  # Standard expansion ratio
        hidden_dim = (hidden_dim + 63) // 64 * 64  # Round to multiple of 64
        
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.ffn = FeedForward(config)
    
    def forward(self, x, rope_cache):
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = x + self.ffn(self.ln_2(x))
        return x


class SimpleGPT(nn.Module):
    """Simple GPT model - clean and debuggable."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (saves memory)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Create RoPE cache
        self.register_buffer('rope_cache', 
            RoPE.create_cos_sin_cache(config.block_size, config.head_dim, config.rope_theta)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print(f"Model initialized: {self.num_parameters():,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def num_parameters(self):
        """Return number of parameters."""
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


def create_model(vocab_size=50257):
    """Create a model with the given vocab size."""
    config = ModelConfig(vocab_size=vocab_size)
    model = SimpleGPT(config)
    return model, config


if __name__ == "__main__":
    # Test the model
    model, config = create_model()
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 100))
    logits, loss = model(x, targets=x)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")