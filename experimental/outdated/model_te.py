#!/usr/bin/env python3
"""
TransformerEngine FP8 version of the GPT model.
Drop-in replacement with FP8 acceleration for compute-heavy layers.
Keeps embeddings and LM head in BF16 for stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# TransformerEngine imports
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    # Try to import MXFP8 for Blackwell GPUs
    try:
        from transformer_engine.common.recipe import MXFP8BlockScaling
        MXFP8_AVAILABLE = True
    except ImportError:
        MXFP8_AVAILABLE = False
        print("Note: MXFP8BlockScaling not available (needs newer TransformerEngine)")
    TE_AVAILABLE = True
except ImportError:
    print("Warning: TransformerEngine not available. Install with: pip install transformer-engine")
    TE_AVAILABLE = False
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
    fp8_amax_history_len: int = 16  # Start conservative, can increase to 64
    fp8_amax_compute_algo: str = "max"
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_heads == 0
        self.head_dim = self.n_embd // self.n_head
        # RoPE requires even head_dim
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        # FP8 requires dimensions divisible by 16
        assert self.n_embd % 16 == 0, "n_embd must be divisible by 16 for FP8"
        assert self.head_dim % 16 == 0, "head_dim must be divisible by 16 for FP8"


def get_fp8_recipe(config, use_mx=None):
    """Get the FP8 recipe for training.
    
    Args:
        config: Model configuration
        use_mx: Use MXFP8 for Blackwell GPUs. Default False (use standard FP8).
    """
    # IMPORTANT: RTX 5090 supports standard FP8, NOT MXFP8!
    # Even though it's Blackwell, it uses datacenter-style FP8
    if use_mx is None:
        use_mx = False  # Default to standard FP8 which works on RTX 5090
    
    if use_mx and MXFP8_AVAILABLE:
        # MXFP8 - currently doesn't work on RTX 5090
        print("Warning: MXFP8 requested but may not work on consumer GPUs")
        print("Using MXFP8BlockScaling")
        return MXFP8BlockScaling(
            fp8_format=Format.E4M3,
            amax_history_len=config.fp8_amax_history_len,
            amax_compute_algo=config.fp8_amax_compute_algo,
        )
    else:
        # Standard FP8 - WORKS on RTX 5090, H100, H200!
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU: {device_name}")
        print("Using standard DelayedScaling FP8 (RTX 5090 compatible)")
        print("  Note: Attention stays in BF16, Linear layers use FP8")
        return DelayedScaling(
            fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
            amax_history_len=config.fp8_amax_history_len,
            amax_compute_algo=config.fp8_amax_compute_algo,
            # fp8_dpa=False by default - RTX 5090 doesn't support FP8 attention
        )


class RoPE:
    """Rotary Position Embeddings - same as original."""
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
        # Ensure cache is on the right device/dtype
        cos_cache = cos_sin_cache[:seq_len, :, 0].to(device=x.device, dtype=x.dtype)
        sin_cache = cos_sin_cache[:seq_len, :, 1].to(device=x.device, dtype=x.dtype)
        
        # Apply rotation
        x_rot = torch.stack([
            x[..., 0] * cos_cache.unsqueeze(0).unsqueeze(2) - x[..., 1] * sin_cache.unsqueeze(0).unsqueeze(2),
            x[..., 0] * sin_cache.unsqueeze(0).unsqueeze(2) + x[..., 1] * cos_cache.unsqueeze(0).unsqueeze(2)
        ], dim=-1)
        
        return x_rot.reshape(batch, seq_len, n_heads, head_dim)


class Attention(nn.Module):
    """Multi-Head Attention with GQA - using TE Linear for FP8 projections, BF16 for attention."""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        
        # Use TransformerEngine Linear for FP8 support
        # Note: TE Linear expects (in_features, out_features) order
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
        
        # QKV projections (now using FP8 via TE)
        # Note: TE requires feature dims (C) divisible by 16, which we ensure in config
        # Sequence length (T) padding is not required for FP8
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = RoPE.apply_rotary_pos_emb(q, rope_cache)
        k = RoPE.apply_rotary_pos_emb(k, rope_cache)
        
        # Reshape for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA: Repeat KV heads
        if self.n_kv_heads != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        
        # Use PyTorch SDPA for attention (stays in BF16)
        # RTX 5090 doesn't support FP8 fused attention kernels
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.dropout_p,
            is_causal=True
        )
        
        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection (FP8)
        y = self.o_proj(y)
        y = self.dropout(y)
        
        return y


class FeedForward(nn.Module):
    """SwiGLU feedforward network - using TE Linear for FP8."""
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.n_embd * 8/3)
        hidden_dim = (hidden_dim + 63) // 64 * 64  # Round to multiple of 64
        
        # Use TransformerEngine Linear for FP8
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
        # SwiGLU activation (FP8 for projections)
        # Note: TE requires feature dims divisible by 16, not sequence length
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with TE components."""
    
    def __init__(self, config):
        super().__init__()
        # Use TransformerEngine RMSNorm for FP8-friendly normalization
        self.ln_1 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.attn = Attention(config)
        self.ln_2 = te.RMSNorm(config.n_embd, eps=1e-6)
        self.ffn = FeedForward(config)
    
    def forward(self, x, rope_cache):
        x = x + self.attn(self.ln_1(x), rope_cache)
        x = x + self.ffn(self.ln_2(x))
        return x


class SimpleGPT_TE(nn.Module):
    """GPT model with TransformerEngine FP8 support."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Keep embeddings in BF16 (not FP8) for stability
        wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying (conventional direction)
        self.lm_head.weight = wte.weight
        
        # Transformer blocks
        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = te.RMSNorm(config.n_embd, eps=1e-6),  # TE RMSNorm
        ))
        
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
        
        # Report model size
        total_params = self.num_parameters()
        print(f"Model initialized with TransformerEngine FP8 support")
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  FP8 enabled: {config.use_fp8}")
        print(f"  Architecture: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
        print(f"  GQA: {config.n_kv_heads} KV heads ({config.n_head//config.n_kv_heads}x compression)")
        print(f"  Note: FP8 requires Hopper (H100/H200) for speedup, not Ampere (A100)")
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
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
        
        # Token embeddings (BF16, not FP8)
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        
        # Forward through transformer blocks (FP8 happens inside)
        for block in self.transformer.h:
            x = block(x, self.rope_cache)
        
        # Final norm and output (norm is FP8-friendly, head is BF16)
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
        """Generate tokens - inference stays in BF16."""
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


def load_from_bfloat16_checkpoint(model, checkpoint_path):
    """Load BF16 checkpoint into FP8 model."""
    print(f"Loading BF16 checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions don't support weights_only
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        # Handle compiled model prefix if present
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("Removing _orig_mod prefix from checkpoint")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    # Load weights - TE modules have same parameter names
    model.load_state_dict(state_dict, strict=False)  # strict=False for TE metadata
    print("Successfully loaded BF16 weights into FP8 model")
    
    return checkpoint.get('iter_num', 0), checkpoint.get('best_val_loss', float('inf'))


if __name__ == "__main__":
    # Test the model
    if not torch.cuda.is_available():
        print("CUDA not available. FP8 requires GPU.")
        import sys
        sys.exit(1)
    
    config = ModelConfig()
    model = SimpleGPT_TE(config).cuda()
    # Cast model to BF16 (crucial for memory/bandwidth)
    model = model.to(torch.bfloat16)
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 128)).cuda()
    
    # Get FP8 recipe
    fp8_recipe = get_fp8_recipe(config)
    
    # Forward with FP8 (on GPU)
    print("\nTesting forward pass with FP8 on GPU...")
    with te.fp8_autocast(enabled=config.use_fp8, fp8_recipe=fp8_recipe):
        logits, loss = model(x, targets=x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation (inference - no FP8)
    print("\nTesting generation (BF16 inference)...")
    context = torch.zeros((1, 1), dtype=torch.long).cuda()
    generated = model.generate(context, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
    
    print("\n✓ Model test successful!")