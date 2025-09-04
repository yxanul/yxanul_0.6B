#!/usr/bin/env python3
"""
CLEAN MoE TransformerEngine FP8 model - Optimized for <500M params
Built on the CLEAN philosophy: Simple is faster for small models.

Key design choices:
1. First layer dense, rest MoE (proven stability)
2. Efficient top-2 routing with minimal overhead
3. No unnecessary optimizations that slow down small models
4. TransformerEngine Linear for FP8 compatibility
5. Optimized tensor operations for routing

Target: ~475M params with default config
"""

import os
os.environ.setdefault('NVTE_FUSED_ATTN_BACKEND', '1')

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    TE_AVAILABLE = True
except ImportError:
    print("ERROR: TransformerEngine not available!")
    import sys
    sys.exit(1)


@dataclass
class MoEConfig:
    # Base model (matching your config)
    vocab_size: int = 32768  # Using 32k BPE tokenizer
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_heads: int = 3  # GQA: 4x compression
    block_size: int = 4096  # Matching your tokenizer context
    dropout: float = 0.05
    bias: bool = False
    rope_theta: float = 10000.0
    
    # FP8 configuration
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    fp8_pad_for_moe: bool = True  # Pad expert inputs for FP8 alignment
    
    # MoE configuration
    num_experts: int = 8  # 8 experts for ~475M params
    top_k: int = 2  # Top-2 routing for stability
    capacity_factor: float = 1.25  # Conservative capacity
    router_aux_loss_coef: float = 0.01  # Load balancing
    router_z_loss_coef: float = 0.001  # Small z-loss for stability
    router_jitter_noise: float = 0.01  # Small noise during training
    num_dense_layers: int = 1  # First layer dense
    
    # Training stability
    expert_dropout: float = 0.1  # Additional dropout in experts
    
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
    print("Using DelayedScaling FP8 (CLEAN MoE)")
    print("  Note: Routing overhead minimized for small models")
    
    return DelayedScaling(
        fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
        amax_history_len=config.fp8_amax_history_len,
        amax_compute_algo=config.fp8_amax_compute_algo,
    )


class RoPE:
    """Rotary Position Embeddings - unchanged from CLEAN."""
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


class CleanAttention(nn.Module):
    """Multi-Head Attention with GQA - unchanged from CLEAN."""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        
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
    
    def forward(self, x, rope_cache):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        q = RoPE.apply_rotary_pos_emb(q, rope_cache)
        k = RoPE.apply_rotary_pos_emb(k, rope_cache)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.n_kv_heads != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.dropout_p,
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.dropout(y)
        
        return y


class DenseFeedForward(nn.Module):
    """Dense SwiGLU feedforward for first layer(s)."""
    
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
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class OptimizedTop2Router(nn.Module):
    """Optimized Top-2 router with minimal overhead."""
    
    def __init__(self, n_embd, num_experts, capacity_factor=1.25, jitter=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.jitter = jitter
        
        # Router weights in FP8-compatible format
        self.gate = te.Linear(
            n_embd, 
            num_experts, 
            bias=False, 
            params_dtype=torch.bfloat16
        )
    
    def forward(self, x):
        """
        Efficient top-2 routing with load balancing.
        Returns: (dispatch_mask, combine_weights, aux_loss, z_loss)
        """
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)
        
        # Router logits
        logits = self.gate(x_flat)  # [N, E]
        
        # Add noise during training for exploration
        if self.training and self.jitter > 0:
            noise = torch.randn_like(logits) * self.jitter
            logits = logits + noise
        
        # Compute gates (probabilities)
        gates = F.softmax(logits, dim=-1, dtype=torch.float32).to(x.dtype)
        
        # Top-2 selection
        top2_vals, top2_idx = torch.topk(gates, k=2, dim=-1)  # [N, 2]
        
        # Load balancing auxiliary loss
        # Fraction of tokens per expert
        me = gates.mean(dim=0)  # [E]
        ce = torch.zeros_like(me)
        ce.scatter_add_(0, top2_idx[:, 0], torch.ones(N, device=x.device, dtype=ce.dtype) / N)
        aux_loss = (self.num_experts * (me * ce).sum())
        
        # Router z-loss (prevents logit explosion)
        z_loss = (logits.float() ** 2).mean()
        
        # Capacity constraint
        capacity = int(self.capacity_factor * N * 2 / self.num_experts)
        
        # Build dispatch mask efficiently
        dispatch_mask = torch.zeros(N, self.num_experts, 2, device=x.device, dtype=torch.bool)
        combine_weights = torch.zeros(N, self.num_experts, 2, device=x.device, dtype=gates.dtype)
        
        for k in range(2):
            # Count tokens per expert
            expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)
            expert_idx = top2_idx[:, k]
            
            # Efficient batched assignment with capacity
            for e in range(self.num_experts):
                mask = expert_idx == e
                indices = mask.nonzero(as_tuple=True)[0]
                
                # Apply capacity constraint
                if indices.numel() > 0:
                    keep = min(capacity, indices.numel())
                    indices = indices[:keep]
                    dispatch_mask[indices, e, k] = True
                    combine_weights[indices, e, k] = top2_vals[indices, k]
        
        return dispatch_mask, combine_weights, aux_loss, z_loss


class MoEExpert(nn.Module):
    """Single SwiGLU expert."""
    
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
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MoEFeedForward(nn.Module):
    """Mixture of Experts feedforward layer."""
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.n_embd = config.n_embd
        self.fp8_pad_for_moe = getattr(config, 'fp8_pad_for_moe', True)
        
        # Router
        self.router = OptimizedTop2Router(
            config.n_embd, 
            config.num_experts,
            config.capacity_factor,
            config.router_jitter_noise if hasattr(config, 'router_jitter_noise') else 0.01
        )
        
        # Experts
        self.experts = nn.ModuleList([
            MoEExpert(config) for _ in range(config.num_experts)
        ])
        
        self.dropout = nn.Dropout(config.expert_dropout if hasattr(config, 'expert_dropout') else config.dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [B*T, C]
        
        # Route tokens
        dispatch_mask, combine_weights, aux_loss, z_loss = self.router(x)
        
        # Process through experts
        output = torch.zeros_like(x_flat)
        
        for e, expert in enumerate(self.experts):
            # Get tokens for this expert (both top-1 and top-2)
            expert_mask = dispatch_mask[:, e, :].any(dim=-1)
            if not expert_mask.any():
                continue
                
            expert_input = x_flat[expert_mask]
            num_tokens = expert_input.shape[0]
            
            # FP8 alignment: ensure batch dimension is divisible by 8
            # Pad if necessary for FP8 execution
            padded = False
            if self.fp8_pad_for_moe and num_tokens % 8 != 0:
                pad_size = 8 - (num_tokens % 8)
                expert_input = torch.cat([
                    expert_input,
                    torch.zeros(pad_size, expert_input.shape[1], 
                               device=expert_input.device, dtype=expert_input.dtype)
                ], dim=0)
                padded = True
            
            expert_output = expert(expert_input)
            
            # Remove padding if we added any
            if padded:
                expert_output = expert_output[:num_tokens]
            
            # Combine with weights - ensure dtype match
            weights = combine_weights[expert_mask, e, :].sum(dim=-1, keepdim=True)
            output[expert_mask] = output[expert_mask] + (expert_output * weights).to(output.dtype)
        
        output = output.view(B, T, C)
        output = self.dropout(output)
        
        return output, aux_loss, z_loss


class MoEBlock(nn.Module):
    """Transformer block with MoE or Dense FFN."""
    
    def __init__(self, config, use_moe=True):
        super().__init__()
        self.ln_1 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = CleanAttention(config)
        self.ln_2 = te.RMSNorm(config.n_embd, eps=1e-6)
        
        self.use_moe = use_moe
        if use_moe:
            self.ffn = MoEFeedForward(config)
        else:
            self.ffn = DenseFeedForward(config)
    
    def forward(self, x, rope_cache):
        # Attention
        x = x + self.attn(self.ln_1(x), rope_cache)
        
        # FFN
        if self.use_moe:
            ffn_out, aux_loss, z_loss = self.ffn(self.ln_2(x))
            x = x + ffn_out
            return x, aux_loss, z_loss
        else:
            x = x + self.ffn(self.ln_2(x))
            return x, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)


class CleanMoE_TE(nn.Module):
    """
    CLEAN MoE model with TransformerEngine FP8.
    
    Achieves <500M params with efficient routing.
    First layer dense for stability, rest MoE.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings (BF16, weight-tied)
        wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = wte.weight  # Weight tying
        
        # Transformer blocks
        blocks = []
        for i in range(config.n_layer):
            # First n layers dense, rest MoE
            use_moe = i >= config.num_dense_layers
            blocks.append(MoEBlock(config, use_moe=use_moe))
        
        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(blocks),
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
        active_params = self.active_parameters()
        print(f"CLEAN MoE model initialized with TransformerEngine FP8")
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  Active parameters per token: ~{active_params:,} ({active_params/1e6:.1f}M)")
        print(f"  Architecture: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
        print(f"  MoE: {config.num_experts} experts, top-{config.top_k}")
        print(f"  Dense layers: first {config.num_dense_layers}")
        print(f"  GQA: {config.n_kv_heads} KV heads ({config.n_head//config.n_kv_heads}x compression)")
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def num_parameters(self):
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def active_parameters(self):
        """Approximate active parameters per forward pass."""
        # Base: embeddings, attention, layer norms
        base_params = sum(p.numel() for n, p in self.named_parameters() 
                         if 'experts' not in n and p.requires_grad)
        
        # Add active experts (2 per layer for MoE layers)
        moe_layers = self.config.n_layer - self.config.num_dense_layers
        if moe_layers > 0:
            single_expert = sum(p.numel() for n, p in self.named_parameters() 
                              if 'experts.0' in n and p.requires_grad) // self.config.num_experts
            base_params += single_expert * 2 * moe_layers  # top-2
        
        return base_params
    
    def forward(self, idx, targets=None):
        """
        CLEAN forward pass with MoE.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        # Token embeddings
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        
        # Accumulate auxiliary losses
        total_aux_loss = 0.0
        total_z_loss = 0.0
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x, aux_loss, z_loss = block(x, self.rope_cache)
            total_aux_loss = total_aux_loss + aux_loss
            total_z_loss = total_z_loss + z_loss
        
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
            
            # Add auxiliary losses
            if self.config.router_aux_loss_coef > 0:
                loss = loss + self.config.router_aux_loss_coef * total_aux_loss
            if self.config.router_z_loss_coef > 0:
                loss = loss + self.config.router_z_loss_coef * total_z_loss
        
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
    # Test the CLEAN MoE model
    if not torch.cuda.is_available():
        print("CUDA not available. FP8 requires GPU.")
        import sys
        sys.exit(1)
    
    config = MoEConfig()
    model = CleanMoE_TE(config).cuda()
    model = model.to(torch.bfloat16)
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 128)).cuda()
    fp8_recipe = get_fp8_recipe(config)
    
    print("\nTesting CLEAN MoE implementation...")
    
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        logits, loss = model(x, targets=x)
    
    print(f"✓ MoE model test successful!")
    print(f"Loss: {loss.item():.4f}")
    print("\nCLEAN MoE: Efficient routing for small-scale models.")