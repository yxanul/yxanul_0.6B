"""
FP8-Optimized Yxanul Model using NVIDIA Transformer Engine
Provides 2x throughput on RTX 4090 with automatic mixed precision in FP8.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import math

# Import Transformer Engine
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("Warning: Transformer Engine not available, falling back to standard implementation")

@dataclass
class ModelConfig:
    """Model configuration with FP8 support."""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    max_position_embeddings: int = 2048
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True
    # FP8 specific
    use_fp8: bool = True
    fp8_margin: int = 0  # Margin for FP8 dynamic scaling
    fp8_interval: int = 1  # Interval for FP8 scaling factor update
    fp8_amax_history_len: int = 16  # History length for FP8 scaling
    fp8_amax_compute_algo: str = "most_recent"  # Algorithm for computing amax


class FP8TransformerBlock(nn.Module):
    """Transformer block using Transformer Engine for FP8 computation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        if TRANSFORMER_ENGINE_AVAILABLE and config.use_fp8:
            # Use Transformer Engine layers for FP8
            self.ln1 = te.LayerNormLinear(
                config.hidden_size,
                config.hidden_size,
                eps=config.layer_norm_eps,
                bias=False,
                return_bias=False
            )
            
            self.attention = te.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                attention_dropout=config.attention_probs_dropout_prob,
                layer_number=None,  # Will be set later
                attn_mask_type="causal"
            )
            
            self.ln2 = te.LayerNormLinear(
                config.hidden_size,
                config.hidden_size,
                eps=config.layer_norm_eps,
                bias=False,
                return_bias=False
            )
            
            # MLP using Transformer Engine
            self.mlp = te.LayerNormMLP(
                config.hidden_size,
                config.intermediate_size,
                eps=config.layer_norm_eps,
                activation='gelu',
                bias=False,
                return_bias=False
            )
        else:
            # Fallback to standard PyTorch layers
            self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.attention = StandardAttention(config)
            self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.mlp = StandardMLP(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        # Self-attention with residual
        if TRANSFORMER_ENGINE_AVAILABLE and self.config.use_fp8:
            # Transformer Engine handles FP8 conversion internally
            residual = hidden_states
            hidden_states = self.ln1(hidden_states)
            attn_output = self.attention(hidden_states, attention_mask=attention_mask)
            hidden_states = residual + self.dropout(attn_output)
            
            # MLP with residual
            residual = hidden_states
            hidden_states = self.ln2(hidden_states)
            mlp_output = self.mlp(hidden_states)
            hidden_states = residual + self.dropout(mlp_output)
        else:
            # Standard forward pass
            residual = hidden_states
            hidden_states = self.ln1(hidden_states)
            attn_output, _ = self.attention(hidden_states, attention_mask)
            hidden_states = residual + self.dropout(attn_output)
            
            residual = hidden_states
            hidden_states = self.ln2(hidden_states)
            mlp_output = self.mlp(hidden_states)
            hidden_states = residual + self.dropout(mlp_output)
        
        return hidden_states, None  # No cache in FP8 mode initially


class StandardAttention(nn.Module):
    """Fallback standard attention when Transformer Engine is not available."""
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # QKV projections
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for attention
        query_layer = query_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        
        query_layer = query_layer.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)
        
        # Output projection
        output = self.out_proj(context_layer)
        
        return output, attention_probs


class StandardMLP(nn.Module):
    """Fallback standard MLP when Transformer Engine is not available."""
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class YxanulFP8Model(nn.Module):
    """Yxanul model with FP8 optimization via Transformer Engine."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            FP8TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Set layer numbers for Transformer Engine
        if TRANSFORMER_ENGINE_AVAILABLE and config.use_fp8:
            for i, block in enumerate(self.blocks):
                if hasattr(block.attention, 'layer_number'):
                    block.attention.layer_number = i
        
        # Final layer norm
        if TRANSFORMER_ENGINE_AVAILABLE and config.use_fp8:
            self.ln_f = te.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # FP8 recipe for training
        if TRANSFORMER_ENGINE_AVAILABLE and config.use_fp8:
            self.fp8_recipe = recipe.DelayedScaling(
                margin=config.fp8_margin,
                interval=config.fp8_interval,
                fp8_format=recipe.Format.HYBRID,  # E4M3 for forward, E5M2 for backward
                amax_history_len=config.fp8_amax_history_len,
                amax_compute_algo=config.fp8_amax_compute_algo
            )
        else:
            self.fp8_recipe = None
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ):
        batch_size, seq_length = input_ids.shape
        
        # Get position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Create causal mask if needed
        if attention_mask is None and seq_length > 1:
            attention_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=input_ids.device) * float('-inf'),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
        
        # Apply transformer blocks with FP8 context
        if TRANSFORMER_ENGINE_AVAILABLE and self.config.use_fp8 and self.fp8_recipe:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                for block in self.blocks:
                    hidden_states, _ = block(hidden_states, attention_mask, use_cache)
        else:
            for block in self.blocks:
                hidden_states, _ = block(hidden_states, attention_mask, use_cache)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }


def create_fp8_model(config_dict: dict = None) -> YxanulFP8Model:
    """Create an FP8-optimized model instance."""
    if config_dict is None:
        config = ModelConfig()
    else:
        config = ModelConfig(**config_dict)
    
    model = YxanulFP8Model(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Created Yxanul FP8 Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  FP8 enabled: {TRANSFORMER_ENGINE_AVAILABLE and config.use_fp8}")
    
    return model