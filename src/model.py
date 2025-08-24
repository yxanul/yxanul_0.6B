"""
Yxanul 0.6B Deep Transformer Model
Implements a deep, narrow transformer with all modern optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass

# Try to import optimized modules
try:
    # Flash Attention 3 has different imports
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
    FLASH_ATTN_VERSION = 3
except ImportError:
    try:
        # Fallback to Flash Attention 2
        from flash_attn import flash_attn_func
        FLASH_ATTN_AVAILABLE = True
        FLASH_ATTN_VERSION = 2
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
        FLASH_ATTN_VERSION = 0
        print("Warning: Flash Attention not available, using standard attention")

try:
    from apex.normalization import FusedLayerNorm
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    FusedLayerNorm = nn.LayerNorm


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Faster than LayerNorm, used in Llama models.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)  # Ensure eps is a float
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class FactorizedEmbedding(nn.Module):
    """Factorized embeddings to reduce parameter count.
    
    Instead of vocab_size × d_model parameters, uses:
    vocab_size × r + r × d_model parameters.
    
    For r=128: 200005×128 + 128×768 = 25.7M vs 153.6M (83% reduction!)
    """
    
    def __init__(self, vocab_size: int = 200005, d_model: int = 768, r: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.r = r
        
        # Factorized components
        self.embed = nn.Embedding(vocab_size, r)
        self.proj = nn.Linear(r, d_model, bias=False)
        
        # Scaled initialization for SuperBPE's larger vocabulary
        scale = math.log(max(2, vocab_size / 50257))  # Scale factor for larger vocab
        self.embed.weight.data.normal_(0, 1.0 / math.sqrt(r * scale))
        self.proj.weight.data.normal_(0, math.sqrt(2.0 / r))
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: embed then project."""
        embedded = self.embed(input_ids)  # (B, L) -> (B, L, r)
        return self.proj(embedded)  # (B, L, r) -> (B, L, d_model)


@dataclass
class ModelConfig:
    """Configuration for the deep transformer model."""
    hidden_size: int = 768
    num_layers: int = 28
    num_attention_heads: int = 12
    num_kv_heads: int = 2  # For Grouped-Query Attention (GQA)
    intermediate_size: int = 2048  # Optimized for SwiGLU: 8/3 * hidden_size
    vocab_size: int = 200005  # SuperBPE t=180k vocabulary
    max_position_embeddings: int = 4096
    
    # Advanced features
    use_swiglu: bool = True
    use_rope: bool = True
    rope_theta: float = 10000.0
    pre_norm: bool = True
    use_rmsnorm: bool = True  # Use RMSNorm instead of LayerNorm
    
    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Initialization
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    
    # Optimizations
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False
    
    # Factorized embeddings
    use_factorized_embeddings: bool = True
    factorization_dim: int = 128  # Bottleneck dimension for factorized embeddings
    

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim, max_position_embeddings=4096, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin
        self._set_cos_sin_cache(max_position_embeddings)
        
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
            
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function - more efficient than GEGLU.
    Used in PaLM, LLaMA, and other modern models.
    """
    
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # Two separate projections for better parameter efficiency
        self.w1 = nn.Linear(dim_in, dim_out, bias=False)
        self.w2 = nn.Linear(dim_in, dim_out, bias=False)
        self.w3 = nn.Linear(dim_out, dim_in, bias=False)
        
    def forward(self, x):
        # SwiGLU: x * swish(gate) where swish(x) = x * sigmoid(x)
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    """Grouped-Query Attention with RoPE and Flash Attention support.
    
    Uses fewer KV heads than Q heads for significant memory and speed improvements.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.use_flash_attention = config.use_flash_attention and FLASH_ATTN_AVAILABLE
        
        # Separate projections for Q and KV (GQA optimization)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, 2 * self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Rotary embeddings
        if config.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )
        else:
            self.rotary_emb = None
            
        self.attention_dropout = config.attention_dropout
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q projection
        q = self.q_proj(hidden_states)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        # KV projection (fewer heads)
        kv = self.kv_proj(hidden_states)
        kv = kv.reshape(batch_size, seq_len, 2, self.num_kv_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, batch, num_kv_heads, seq_len, head_dim)
        k, v = kv[0], kv[1]
        
        # Apply RoPE
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(hidden_states, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat KV heads to match Q heads (GQA)
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Attention computation
        if self.use_flash_attention:
            # Flash Attention (much faster)
            if FLASH_ATTN_VERSION >= 3:
                # Flash Attention 3 - use QKV packed format for better performance
                # Combine QKV for packed format
                qkv = torch.stack([q, k, v], dim=2)  # (batch, num_heads, 3, seq_len, head_dim)
                qkv = qkv.transpose(1, 3).contiguous()  # (batch, seq_len, 3, num_heads, head_dim)
                
                attn_output = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    causal=True,
                    window_size=(-1, -1),  # Full attention window
                    alibi_slopes=None
                )
            else:
                # Flash Attention 2 fallback
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    causal=True
                )
            attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        else:
            # Standard attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        if config.use_swiglu:
            # SwiGLU uses custom module
            self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)
        else:
            # Standard FFN
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
            self.act = nn.GELU()
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
            
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.use_swiglu = config.use_swiglu
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            hidden_states = self.ffn(hidden_states)
        else:
            hidden_states = self.up_proj(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.down_proj(hidden_states)
            
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single transformer layer with pre-norm and all optimizations."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Choose normalization layer
        if config.use_rmsnorm:
            NormLayer = RMSNorm
        else:
            NormLayer = FusedLayerNorm if APEX_AVAILABLE else nn.LayerNorm
        
        # Create normalization layers
        self.ln_1 = NormLayer(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = NormLayer(config.hidden_size, eps=config.layer_norm_eps)
        
        # Core components
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        
        # Whether to use pre-normalization
        self.pre_norm = config.pre_norm
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        
        if use_checkpoint:
            # Gradient checkpointing for memory efficiency
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                hidden_states,
                attention_mask,
            )
        else:
            return self._forward(hidden_states, attention_mask)
    
    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm architecture (more stable)
        if self.pre_norm:
            # Attention block
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            hidden_states = self.attention(hidden_states, attention_mask)
            hidden_states = residual + hidden_states
            
            # FFN block
            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            hidden_states = self.feed_forward(hidden_states)
            hidden_states = residual + hidden_states
        else:
            # Post-norm (traditional)
            residual = hidden_states
            hidden_states = self.attention(hidden_states, attention_mask)
            hidden_states = residual + hidden_states
            hidden_states = self.ln_1(hidden_states)
            
            residual = hidden_states
            hidden_states = self.feed_forward(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = self.ln_2(hidden_states)
            
        return hidden_states


class YxanulModel(nn.Module):
    """The main Yxanul 0.6B deep transformer model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (factorized if enabled)
        if config.use_factorized_embeddings:
            self.embed_tokens = FactorizedEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.factorization_dim
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Dropout (removed learned position embeddings - RoPE is sufficient)
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Cache for causal mask to avoid recreation
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((config.max_position_embeddings, config.max_position_embeddings), float('-inf')),
                diagonal=1
            )
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        if config.use_rmsnorm:
            self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            LayerNorm = FusedLayerNorm if APEX_AVAILABLE else nn.LayerNorm
            self.ln_f = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Scale initialization for deep networks
        for layer_idx, layer in enumerate(self.layers):
            # Scale by depth
            scale = math.sqrt(2.0 * config.num_layers)
            layer.attention.o_proj.weight.data.div_(scale)
            # For SwiGLU, scale w3 (down projection), otherwise scale down_proj
            if config.use_swiglu:
                layer.feed_forward.ffn.w3.weight.data.div_(scale)
            else:
                layer.feed_forward.down_proj.weight.data.div_(scale)
            
    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeddings = self.embed_tokens(input_ids)
        
        # Apply token embeddings (position encoding handled by RoPE in attention)
        hidden_states = token_embeddings
        hidden_states = self.dropout(hidden_states)
        
        # Use cached causal mask
        if attention_mask is None:
            attention_mask = self.causal_mask[:seq_len, :seq_len]
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply transformer layers
        checkpoint_every_n = 4  # Checkpoint every 4th layer
        for i, layer in enumerate(self.layers):
            # Selective gradient checkpointing - checkpoint every Nth layer
            use_checkpoint = (
                self.config.use_gradient_checkpointing
                and self.training
                and (i + 1) % checkpoint_every_n == 0  # Fixed: was checking i % 4 == 0
            )
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_checkpoint=use_checkpoint
            )
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states


class YxanulForCausalLM(nn.Module):
    """Yxanul model with language modeling head."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = YxanulModel(config)
        
        # For factorized embeddings, we don't need a separate lm_head
        if not config.use_factorized_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            # Properly tie embeddings
            self.tie_weights()
        else:
            self.lm_head = None
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.config.use_gradient_checkpointing = True
        if hasattr(self.model, 'config'):
            self.model.config.use_gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.config.use_gradient_checkpointing = False
        if hasattr(self.model, 'config'):
            self.model.config.use_gradient_checkpointing = False
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # Get model outputs
        hidden_states = self.model(input_ids, attention_mask)
        
        # Language modeling head
        if self.config.use_factorized_embeddings:
            # Efficient two-matmul computation for factorized embeddings
            # First: hidden_states @ proj.weight.T -> (B, L, r)
            # Second: (B, L, r) @ embed.weight.T -> (B, L, vocab_size)
            # Note: F.linear expects weight, not weight.T
            h_proj = F.linear(hidden_states, self.model.embed_tokens.proj.weight.T)  # (B,L,768) -> (B,L,128)
            logits = F.linear(h_proj, self.model.embed_tokens.embed.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
        return (loss, logits) if loss is not None else logits
    
    def tie_weights(self):
        """Tie the weights between token embeddings and lm_head."""
        # Only tie weights for non-factorized embeddings
        if not self.config.use_factorized_embeddings:
            # CORRECT weight tying - share the SAME tensor object, not a copy!
            if self.lm_head.weight.shape == self.model.embed_tokens.weight.shape:
                self.lm_head.weight = self.model.embed_tokens.weight  # Share the SAME object
            else:
                print(f"Warning: Cannot tie weights, shapes don't match: "
                      f"lm_head: {self.lm_head.weight.shape}, "
                      f"embed_tokens: {self.model.embed_tokens.weight.shape}")


def create_model(config_dict: dict = None) -> YxanulForCausalLM:
    """Create a Yxanul model with the given configuration."""
    if config_dict is None:
        config = ModelConfig()
    else:
        config = ModelConfig(**config_dict)
    
    model = YxanulForCausalLM(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params / 1e6:.1f}M parameters")
    print(f"Layers: {config.num_layers}, Hidden size: {config.hidden_size}")
    print(f"Attention: {config.num_attention_heads} Q heads, {config.num_kv_heads} KV heads (GQA ratio {config.num_attention_heads//config.num_kv_heads}:1)")
    print(f"Flash Attention: {config.use_flash_attention and FLASH_ATTN_AVAILABLE}")
    print(f"Using SwiGLU: {config.use_swiglu}, RMSNorm: {config.use_rmsnorm}")
    
    return model