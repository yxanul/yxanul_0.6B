#!/usr/bin/env python3
"""
Yxanul Model using TransformerEngine v2.4 Native Modules

This implementation leverages TE v2.4's optimized TransformerLayer which includes:
- Native GQA (Grouped Query Attention) support
- Native RMSNorm implementation  
- Native SwiGLU activation
- Fused QKV operations
- Automatic FP8 mixed precision
- Built-in RoPE support

Expected performance: 40-50% faster than manual implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math

# TransformerEngine v2.4 imports
try:
    import transformer_engine as te
    import transformer_engine.pytorch as te_pytorch
    from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
    from transformer_engine.common.recipe import DelayedScaling, Format
    TE_AVAILABLE = True
    print(f"TransformerEngine v{te.__version__} loaded successfully")
except ImportError:
    TE_AVAILABLE = False
    print("WARNING: TransformerEngine not found. Install from NGC container for optimal performance.")


@dataclass
class ModelConfig:
    """Configuration for Yxanul TE v2.4 model"""
    # Model dimensions
    vocab_size: int = 200005  # SuperBPE tokenizer
    hidden_size: int = 640  # 270M model default
    intermediate_size: int = 1708  # ~2.67x hidden
    num_hidden_layers: int = 28
    
    # Attention configuration
    num_attention_heads: int = 10  # 10 heads * 64 = 640
    num_kv_heads: int = 3  # For GQA with ~3:1 ratio
    head_dim: int = 64  # hidden_size // num_attention_heads
    
    # Positional encoding
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    
    # Regularization
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Precision
    use_fp8: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    
    # Optimizations
    fuse_qkv_params: bool = True
    use_flash_attention: bool = True
    
    # Factorization for embeddings (Gemma-style)
    use_factorized_embedding: bool = True
    factorization_dim: int = 128
    
    # Normalization
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False


class FactorizedEmbedding(nn.Module):
    """Factorized embedding layer for parameter efficiency"""
    
    def __init__(self, vocab_size: int, hidden_size: int, factorization_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.factorization_dim = factorization_dim
        
        # Two-step embedding: vocab -> factor_dim -> hidden_size
        self.embed = nn.Embedding(vocab_size, factorization_dim)
        
        # Use TE Linear for potential FP8 optimization
        if TE_AVAILABLE:
            self.proj = te_pytorch.Linear(
                factorization_dim, 
                hidden_size,
                bias=False,
                params_dtype=torch.bfloat16
            )
        else:
            self.proj = nn.Linear(factorization_dim, hidden_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        nn.init.normal_(self.embed.weight, 0, 0.02)
        nn.init.normal_(self.proj.weight, 0, 0.02)  # Standard LLM initialization
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.proj(x)
        return x


class YxanulTEv2Model(nn.Module):
    """
    Yxanul model using TransformerEngine v2.4 native modules.
    
    This model leverages TE's TransformerLayer which internally handles:
    - Attention with GQA support
    - FFN with SwiGLU activation
    - RMSNorm normalization
    - FP8 mixed precision
    - Fused operations
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False  # Can be enabled to save memory
        
        if not TE_AVAILABLE:
            raise RuntimeError("TransformerEngine is required for this model. Install from NGC container.")
        
        # Token embeddings (factorized for efficiency)
        if config.use_factorized_embedding:
            self.embed_tokens = FactorizedEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.factorization_dim
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Build transformer layers using TE v2.4 TransformerLayer
        self.layers = nn.ModuleList()
        
        # Initialize RoPE (TE v2.4 way)
        self.rope = RotaryPositionEmbedding(config.head_dim)
        
        for layer_idx in range(config.num_hidden_layers):
            layer = te_pytorch.TransformerLayer(
                # Core dimensions
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.intermediate_size,
                
                # Attention configuration
                num_attention_heads=config.num_attention_heads,
                num_gqa_groups=config.num_kv_heads,  # Native GQA support!
                kv_channels=config.head_dim,
                
                # Activations and normalization
                activation='swiglu',  # Native SwiGLU
                normalization='RMSNorm',  # Native RMSNorm
                layernorm_epsilon=config.layernorm_epsilon,
                
                # Layer identification
                layer_number=layer_idx + 1,
                
                # Attention settings
                self_attn_mask_type='padding_causal',  # Support both padding and causal masking
                attention_dropout=config.attention_dropout,
                hidden_dropout=config.hidden_dropout,
                
                # Optimization flags
                bias=False,  # No bias for better training stability
                fuse_qkv_params=config.fuse_qkv_params,  # Fused QKV
                params_dtype=config.params_dtype,
                
                # Position encoding - RoPE is handled separately in TE v2.4
                seq_length=config.max_position_embeddings,
                # Note: apply_rotary_pos_emb and rotary_percent removed for TE v2.4
                
                # Parallelism (single GPU for now)
                tp_size=1,
                tp_group=None,
                sequence_parallel=False,
                
                # Note: No layer_type for decoder-only models (default is self-attention)
                
                # FP8 settings
                zero_centered_gamma=config.zero_centered_gamma,
                
                # Memory optimization
                # Set to False - True requires Megatron-style main_grad attribute
                fuse_wgrad_accumulation=False
                
                # Note: attention_type removed - not valid in TE v2.4
                # TE auto-selects best backend: Flash Attention 3, Fused, or Unfused
            )
            self.layers.append(layer)
        
        # Final RMSNorm using TE's implementation
        self.final_norm = te_pytorch.RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            zero_centered_gamma=config.zero_centered_gamma
        )
        
        # Language modeling head (keep in PyTorch for stability)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if using factorized embeddings
        if not config.use_factorized_embedding:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self._init_weights()
        
        # Print model info
        self._print_model_info()
    
    def _init_weights(self):
        """Initialize weights for non-TE modules"""
        # LM head initialization
        nn.init.normal_(self.lm_head.weight, 0, 0.02)
        
        # Embedding initialization (if not factorized)
        if not self.config.use_factorized_embedding:
            nn.init.normal_(self.embed_tokens.weight, 0, 0.02)
    
    def _print_model_info(self):
        """Print model configuration and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("Yxanul TE v2.4 Model Configuration")
        print("="*60)
        print(f"Total parameters: {total_params/1e6:.1f}M")
        print(f"Trainable parameters: {trainable_params/1e6:.1f}M")
        print(f"Hidden size: {self.config.hidden_size}")
        print(f"Layers: {self.config.num_hidden_layers}")
        print(f"Attention heads: {self.config.num_attention_heads}")
        print(f"KV heads (GQA): {self.config.num_kv_heads}")
        print(f"GQA ratio: {self.config.num_attention_heads // self.config.num_kv_heads}:1")
        print(f"Activation: SwiGLU (native)")
        print(f"Normalization: RMSNorm (native)")
        print(f"FP8 enabled: {self.config.use_fp8}")
        print(f"Factorized embedding: {self.config.use_factorized_embedding}")
        print("="*60 + "\n")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling loss
            use_cache: Whether to return key-value cache (not implemented)
            return_dict: Whether to return a dictionary (not implemented)
        
        Returns:
            (loss, logits) if labels provided, else just logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Ensure sequence length is multiple of 16 for FP8 optimization
        original_seq_len = seq_len
        if self.config.use_fp8 and seq_len % 16 != 0:
            pad_len = 16 - (seq_len % 16)
            input_ids = F.pad(input_ids, (0, pad_len), value=0)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            if labels is not None:
                labels = F.pad(labels, (0, pad_len), value=-100)
            seq_len = input_ids.shape[1]
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Generate RoPE frequencies for EXACT sequence length
        # TE v2.4+ requires exact match - no caching/slicing allowed
        freqs_cis = self.rope(seq_len)
        if hidden_states.is_cuda:
            freqs_cis = freqs_cis.to(hidden_states.device)
        
        # Convert HuggingFace-style attention mask to TE format
        # HF: 1 = keep, 0 = mask out
        # TE: True = mask out, False = keep, shape [B, 1, 1, S]
        te_attention_mask = None
        if attention_mask is not None:
            # Convert: HF uses 1 for keep, TE uses True for masked
            te_attention_mask = ~(attention_mask.bool())  # Invert: 0->True, 1->False
            te_attention_mask = te_attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        
        # Pass through transformer layers
        # TE handles causal masking internally when self_attn_mask_type='causal'
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # TransformerEngine layers don't work well with standard gradient checkpointing
                # Just run normally - TE has its own memory optimizations
                hidden_states = layer(
                    hidden_states,
                    attention_mask=te_attention_mask,
                    rotary_pos_emb=freqs_cis  # Pass RoPE frequencies (TE v2.4 API)
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=te_attention_mask,
                    rotary_pos_emb=freqs_cis  # Pass RoPE frequencies (TE v2.4 API)
                )
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Remove padding if we added it
        if original_seq_len != seq_len:
            logits = logits[:, :original_seq_len, :]
            if labels is not None:
                labels = labels[:, :original_seq_len]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten and calculate loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return (loss, logits) if loss is not None else logits


def create_te_v2_model(config: ModelConfig, fp8_recipe=None, enable_gradient_checkpointing=False) -> YxanulTEv2Model:
    """
    Create and initialize a Yxanul TE v2.4 model.
    
    TransformerEngine layers handle FP8 initialization internally,
    so we don't need to call fp8_model_init().
    
    Args:
        config: Model configuration
        fp8_recipe: FP8 recipe to use (for reference, not used here)
        enable_gradient_checkpointing: Enable gradient checkpointing to save memory
    """
    if not TE_AVAILABLE:
        raise RuntimeError("TransformerEngine is required. Please use NGC container.")
    
    # Create the model
    model = YxanulTEv2Model(config)
    
    # Gradient checkpointing disabled - doesn't work well with TE layers
    if enable_gradient_checkpointing:
        print("WARNING: Gradient checkpointing not compatible with TE layers - disabled")
        model.gradient_checkpointing = False
    else:
        model.gradient_checkpointing = False
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to GPU: {torch.cuda.get_device_name()}")
    
    # TransformerEngine layers handle FP8 internally via fp8_autocast
    # No need for fp8_model_init() as shown in official examples
    if config.use_fp8:
        print(f"Model configured for FP8 training (will use fp8_autocast during forward)")
    
    return model


if __name__ == "__main__":
    """Test the model creation and forward pass"""
    
    # Create config
    config = ModelConfig()
    
    # Create model (TransformerEngine layers handle FP8 internally)
    model = create_te_v2_model(config)
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    print("\nTesting forward pass...")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    with torch.no_grad():
        # Test with fp8_autocast
        if config.use_fp8:
            fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max",
                fp8_dpa=False,  # Run attention in BF16
                fp8_mha=False   # Run MHA in BF16
            )
            
            with te_pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                loss, logits = model(input_ids, labels=input_ids)
        else:
            loss, logits = model(input_ids, labels=input_ids)
    
    print(f"✓ Forward pass successful!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")