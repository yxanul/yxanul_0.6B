"""
Optimized FP8 Model for Yxanul 197M
Using DeepSeek's precision strategy without unnecessary overhead.
Simplified for sub-1B parameter models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from functools import lru_cache

# Only import Transformer Engine if available
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    HAS_TE = True
except ImportError:
    HAS_TE = False
    print("Transformer Engine not found. Install from NVIDIA NGC container for FP8 support.")

@dataclass
class FP8Config:
    """FP8 configuration matching DeepSeek's strategy (simplified for small models)"""
    use_fp8: bool = True
    
    # Components that stay in BF16 (DeepSeek strategy)
    embeddings_dtype: str = "bfloat16"
    layernorm_dtype: str = "bfloat16"
    output_head_dtype: str = "bfloat16"
    
    # Components that use FP8
    attention_compute_dtype: str = "fp8_e4m3"
    ffn_compute_dtype: str = "fp8_e4m3"
    
    # Optimizer always FP32
    optimizer_dtype: str = "float32"
    master_weights_dtype: str = "float32"
    
    # Simplified - no block quantization for <1B models
    use_block_quantization: bool = False  # Not worth it for 197M


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better length generalization."""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies for efficiency
        self.register_buffer("freqs_cos", None, persistent=False)
        self.register_buffer("freqs_sin", None, persistent=False)
        self._compute_freqs(max_seq_len)
    
    def _compute_freqs(self, seq_len: int):
        """Compute cosine and sine frequencies for RoPE."""
        # Create position indices
        position = torch.arange(seq_len, dtype=torch.float32)
        
        # Create frequency bands (theta^(-2i/d) for i in [0, d/2))
        dim_half = self.dim // 2
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        
        # Compute angles (position * frequency)
        angles = position.unsqueeze(1) * freqs.unsqueeze(0)  # [seq_len, dim_half]
        
        # Precompute cos and sin
        self.freqs_cos = torch.cos(angles).repeat(1, 2)  # [seq_len, dim]
        self.freqs_sin = torch.sin(angles).repeat(1, 2)  # [seq_len, dim]
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape [batch, heads, seq_len, head_dim]
            k: Key tensor of shape [batch, heads, seq_len, head_dim]
            position_ids: Optional position indices [batch, seq_len]
            
        Returns:
            Rotated query and key tensors
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Ensure we have precomputed frequencies for this sequence length
        if self.freqs_cos is None or self.freqs_cos.shape[0] < seq_len:
            self._compute_freqs(max(seq_len, self.max_seq_len))
            self.freqs_cos = self.freqs_cos.to(q.device)
            self.freqs_sin = self.freqs_sin.to(q.device)
        
        # Get frequencies for current sequence length
        if position_ids is not None:
            # Use provided position IDs (for cases like padding)
            freqs_cos = self.freqs_cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim]
            freqs_sin = self.freqs_sin[position_ids].unsqueeze(1)
        else:
            # Use default sequential positions
            freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]
            freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(1)
        
        # Ensure frequencies are on the same device and dtype as inputs
        freqs_cos = freqs_cos.to(q.device).to(q.dtype)
        freqs_sin = freqs_sin.to(q.device).to(q.dtype)
        
        # Apply rotation using complex number formula: (a + ib) * (cos θ + i sin θ)
        q_rotated = (q * freqs_cos) + (self.rotate_half(q) * freqs_sin)
        k_rotated = (k * freqs_cos) + (self.rotate_half(k) * freqs_sin)
        
        return q_rotated, k_rotated


class OptimizedFP8Embedding(nn.Module):
    """Embeddings stay in BF16 (following DeepSeek)"""
    def __init__(self, vocab_size: int, hidden_size: int, factorization_dim: int = 128):
        super().__init__()
        # Factorized embeddings for memory efficiency
        self.embed = nn.Embedding(vocab_size, factorization_dim)
        self.proj = nn.Linear(factorization_dim, hidden_size, bias=False)
        
        # Keep in BF16 (not FP8)
        self.embed = self.embed.to(torch.bfloat16)
        self.proj = self.proj.to(torch.bfloat16)
        
        # Initialize with scaled values
        scale = math.log(max(2, vocab_size / 50257))
        nn.init.normal_(self.embed.weight, 0, 1.0 / math.sqrt(factorization_dim * scale))
        nn.init.normal_(self.proj.weight, 0, math.sqrt(2.0 / factorization_dim))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Computation stays in BF16
        x = self.embed(input_ids)
        x = self.proj(x)
        return x


class FP8Attention(nn.Module):
    """Attention with FP8 compute, BF16 for sensitive operations, and RoPE"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # GQA configuration - properly parameterized
        self.num_kv_heads = getattr(config, 'num_kv_heads', self.num_heads)  # Default to MHA if not specified
        self.kv_head_dim = self.head_dim  # Same dimension per head
        self.repeat_factor = self.num_heads // self.num_kv_heads  # How many times to repeat KV heads
        
        # Validate GQA configuration
        assert self.num_heads % self.num_kv_heads == 0, f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        
        # Initialize RoPE
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)
        self.rotary_emb = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_position_embeddings,
            theta=rope_theta
        )
        
        if HAS_TE and config.use_fp8:
            # Use Transformer Engine's FP8 Linear layers
            self.q_proj = te.Linear(
                self.hidden_size, 
                self.hidden_size,
                bias=False,
                params_dtype=torch.bfloat16  # Weights stored in BF16
                # FP8 is controlled by fp8_autocast context manager, not here
            )
            self.k_proj = te.Linear(
                self.hidden_size,
                self.num_kv_heads * self.kv_head_dim,  # 2 heads * head_dim
                bias=False,
                params_dtype=torch.bfloat16
            )
            self.v_proj = te.Linear(
                self.hidden_size,
                self.num_kv_heads * self.kv_head_dim,  # 2 heads * head_dim
                bias=False,
                params_dtype=torch.bfloat16
            )
            self.o_proj = te.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                params_dtype=torch.bfloat16
            )
        else:
            # Fallback to BF16
            self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.kv_head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.kv_head_dim, bias=False)
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            
            # Convert to BF16
            for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
                module.to(torch.bfloat16)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projections (FP8 compute if available)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        
        # Apply RoPE to queries and keys (not values, as per standard RoPE)
        q, k = self.rotary_emb(q, k)
        
        # Expand KV heads if using GQA (when num_kv_heads < num_heads)
        if self.repeat_factor > 1:
            # Repeat KV heads to match Q heads for compatibility
            # Note: Modern SDPA kernels can handle GQA natively without expansion
            # but we expand here for compatibility with older kernels
            k = k.repeat_interleave(self.repeat_factor, dim=1)
            v = v.repeat_interleave(self.repeat_factor, dim=1)
        
        # Prepare attention mask if provided (for padding)
        # SDPA expects mask shape [batch, num_heads, seq_len, seq_len] 
        sdpa_mask = None
        if attention_mask is not None:
            # Convert padding mask to attention mask
            # attention_mask shape: [batch_size, seq_len] (1 for real tokens, 0 for padding)
            # Create causal mask first
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            
            # Expand attention_mask for broadcasting
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            padding_mask_expanded = padding_mask.expand(-1, self.num_heads, seq_len, -1)  # [batch, heads, seq_len, seq_len]
            
            # Combine masks: mask if causal OR padding
            combined_mask = causal_mask | (~padding_mask_expanded.bool())
            
            # Convert to float with -inf for masked positions
            sdpa_mask = torch.zeros_like(combined_mask, dtype=q.dtype)
            sdpa_mask[combined_mask] = -float('inf')
        
        # Use scaled_dot_product_attention for better performance and stability
        # This handles numerical stability and can use Flash Attention
        
        # Only use CUDA-specific context on CUDA devices
        if hidden_states.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,  # Use Flash Attention if available
                enable_math=True,   # Fallback to math implementation
                enable_mem_efficient=True  # Use memory-efficient attention
            ):
                # SDPA handles scaling and softmax in a fused kernel
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=sdpa_mask,  # Combined causal + padding mask if provided
                    dropout_p=0.0,
                    is_causal=False if sdpa_mask is not None else True,  # Use explicit mask if available
                    scale=1.0 / math.sqrt(self.head_dim)
                )
        else:
            # CPU or other devices - SDPA will use appropriate implementation
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=False if sdpa_mask is not None else True,
                scale=1.0 / math.sqrt(self.head_dim)
            )
        
        # Reshape and project output (FP8 if available)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output


class FP8FFN(nn.Module):
    """FFN with FP8 compute (bulk of parameters)"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        if HAS_TE and config.use_fp8:
            # SwiGLU with FP8
            self.gate_proj = te.Linear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                params_dtype=torch.bfloat16
            )
            self.up_proj = te.Linear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                params_dtype=torch.bfloat16
            )
            self.down_proj = te.Linear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                params_dtype=torch.bfloat16
            )
        else:
            # BF16 fallback
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            
            for module in [self.gate_proj, self.up_proj, self.down_proj]:
                module.to(torch.bfloat16)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output


class OptimizedTransformerBlock(nn.Module):
    """Transformer block with DeepSeek-style mixed precision"""
    def __init__(self, config):
        super().__init__()
        
        # LayerNorms stay in BF16 (critical for stability)
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-6)  # Match DeepSeek's eps
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        
        # Keep norms in BF16
        self.norm1 = self.norm1.to(torch.bfloat16)
        self.norm2 = self.norm2.to(torch.bfloat16)
        
        # Attention and FFN use FP8 for compute
        self.attention = FP8Attention(config)
        self.ffn = FP8FFN(config)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)  # BF16
        hidden_states = self.attention(hidden_states, attention_mask)  # FP8 compute with mask
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)  # BF16
        hidden_states = self.ffn(hidden_states)  # FP8 compute
        hidden_states = residual + hidden_states
        
        return hidden_states


class YxanulFP8Model(nn.Module):
    """Complete model with optimized FP8 mixed precision"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fp8_config = FP8Config()
        
        # Embeddings (BF16, not FP8)
        self.embeddings = OptimizedFP8Embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            factorization_dim=config.factorization_dim
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            OptimizedTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Output head (BF16, not FP8)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head = self.lm_head.to(torch.bfloat16)
        
        # Final norm (BF16)
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.final_norm = self.final_norm.to(torch.bfloat16)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        fp8_recipe: Optional[Any] = None  # Recipe passed from trainer
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # Trainer controls FP8 context - model just runs normally
        # If fp8_recipe is provided, trainer has already set up the context
        # We don't need brittle detection - trainer is the single source of truth
        
        # Check device for autocast decisions
        device = input_ids.device
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        
        # If fp8_recipe is provided, trainer has set up FP8 context
        # Otherwise, use BF16 autocast on CUDA
        if fp8_recipe is not None:
            # Trainer has set up FP8 context, just run normally
            hidden_states = self.embeddings(input_ids)  # BF16
            
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask)  # Mixed FP8/BF16 with mask
            
            hidden_states = self.final_norm(hidden_states)  # BF16
            logits = self.lm_head(hidden_states)  # BF16
        elif device_type == 'cuda':
            # Use BF16 autocast on CUDA
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                hidden_states = self.embeddings(input_ids)
                
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)
                
                hidden_states = self.final_norm(hidden_states)
                logits = self.lm_head(hidden_states)
        else:
            # CPU path - no autocast
            hidden_states = self.embeddings(input_ids)
            
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask)
            
            hidden_states = self.final_norm(hidden_states)
            logits = self.lm_head(hidden_states)
        
        # Safety guard: Ensure logits are in low precision before loss computation
        assert logits.dtype in (torch.bfloat16, torch.float16, torch.float32), \
            f"Unexpected logits dtype: {logits.dtype}. Expected BF16/FP16/FP32"
        
        loss = None
        if labels is not None:
            # Loss computation in FP32 for stability
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            
            # Cast to FP32 for loss (critical for numerical stability)
            shift_logits = shift_logits.float()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return loss, logits  # Return (loss, logits) to match trainer expectations


def print_precision_summary(model: YxanulFP8Model):
    """Print precision distribution like DeepSeek analysis"""
    fp8_params = 0
    bf16_params = 0
    fp32_params = 0
    
    for name, param in model.named_parameters():
        if 'embed' in name or 'norm' in name or 'lm_head' in name:
            bf16_params += param.numel()
        elif any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate', 'up', 'down']):
            fp8_params += param.numel()
        else:
            fp32_params += param.numel()
    
    total = fp8_params + bf16_params + fp32_params
    
    print("=" * 60)
    print("PRECISION DISTRIBUTION (DeepSeek-style)")
    print("=" * 60)
    print(f"FP8 (compute):  {fp8_params:,} ({fp8_params/total*100:.1f}%)")
    print(f"BF16 (critical): {bf16_params:,} ({bf16_params/total*100:.1f}%)")
    print(f"FP32 (scales):   {fp32_params:,} ({fp32_params/total*100:.1f}%)")
    print(f"Total:           {total:,}")
    print("=" * 60)


# Export ModelConfig for compatibility with train_fp8.py
@dataclass
class ModelConfig:
    """Model configuration compatible with training scripts."""
    vocab_size: int = 200005
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_kv_heads: int = 2
    factorization_dim: int = 128
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    use_fp8: bool = True
    # Additional fields for compatibility
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True


def create_fp8_model(config_dict: dict = None) -> YxanulFP8Model:
    """Create an FP8-optimized model instance."""
    if config_dict is None:
        config = ModelConfig()
    else:
        # Handle both dict and dataclass inputs
        if isinstance(config_dict, dict):
            config = ModelConfig(**config_dict)
        else:
            config = config_dict
    
    model = YxanulFP8Model(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created YxanulFP8Model with {total_params / 1e6:.1f}M parameters")
    
    if HAS_TE and config.use_fp8:
        print("FP8 training enabled via Transformer Engine")
    else:
        print("Using BF16 precision (FP8 not available)")
    
    return model


if __name__ == "__main__":
    # Test configuration
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        vocab_size: int = 200005  # SuperBPE
        hidden_size: int = 768
        intermediate_size: int = 2048
        num_hidden_layers: int = 28
        num_attention_heads: int = 12
        num_kv_heads: int = 2  # GQA with 6:1 ratio
        factorization_dim: int = 128
        max_position_embeddings: int = 4096
        rope_theta: float = 10000.0
        use_fp8: bool = True
    
    config = TestConfig()
    model = YxanulFP8Model(config)
    
    print_precision_summary(model)
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    
    with torch.no_grad():
        logits, loss = model(input_ids)
        print(f"\nForward pass successful!")
        print(f"Logits shape: {logits.shape}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")