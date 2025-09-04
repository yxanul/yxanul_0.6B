"""
Clean V2 MoE implementation based on optimization findings.
Key changes from V1:
- No base MLP (2x speedup)
- Switch Transformer routing with capacity limits
- Larger experts (3-4x expansion)
- Comprehensive monitoring built-in
- No MTP for simplicity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math

# Check for Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Flash Attention not available, using memory-efficient fallback")


@dataclass
class MoEConfig:
    """Configuration for V2 MoE model."""
    # Model dimensions
    n_embd: int = 896
    n_head: int = 28
    n_kv_head: int = 7  # GQA with 4x reduction
    n_layer: int = 24
    block_size: int = 2048
    vocab_size: int = 32768  # 32k BPE tokenizer
    
    # MoE configuration
    num_experts: int = 4  # 2-4 experts as recommended
    expert_expansion: float = 3.5  # Larger since no base MLP
    capacity_factor: float = 1.0  # Start conservative, can increase to 1.25-1.5
    
    # Routing
    router_type: str = "switch"  # Switch Transformer routing
    router_aux_loss_weight: float = 0.01  # Load balancing loss
    router_z_loss_weight: float = 0.001  # Prevent logit explosion
    
    # Overflow handling
    overflow_policy: str = "drop"  # "drop" or "rescue"
    rescue_expansion: float = 1.0  # Small rescue FFN if overflow_policy="rescue"
    
    # Training
    dropout: float = 0.1
    bias: bool = False
    
    # Monitoring
    log_interval: int = 100  # Log expert stats every N steps
    
    def __post_init__(self):
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head for GQA"
        self.head_dim = self.n_embd // self.n_head


class SwitchRouter(nn.Module):
    """
    Switch Transformer router with capacity-based load balancing.
    Each expert processes at most C = ceil(batch_tokens * capacity_factor / num_experts) tokens.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.capacity_factor = config.capacity_factor
        
        # Router network
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        
        # Tracking
        self.register_buffer("expert_counts", torch.zeros(config.num_experts))
        self.register_buffer("total_tokens", torch.tensor(0, dtype=torch.long))
        self.register_buffer("dropped_tokens", torch.tensor(0, dtype=torch.long))
    
    def compute_capacity(self, batch_tokens: int) -> int:
        """Compute per-expert capacity."""
        return math.ceil(batch_tokens * self.capacity_factor / self.num_experts)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        batch_tokens = B * T
        
        # Compute router logits and probabilities
        router_logits = self.router(x)  # [B, T, E]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-1 expert selection
        expert_indices = router_probs.argmax(dim=-1)  # [B, T]
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()  # [B, T, E]
        
        # Compute expert capacity
        capacity = self.compute_capacity(batch_tokens)
        
        # Track how many tokens go to each expert
        tokens_per_expert = expert_mask.sum(dim=(0, 1))  # [E]
        
        # Create dispatch mask (which tokens actually get processed)
        # This implements the capacity constraint
        dispatch_mask = torch.zeros_like(expert_mask)
        
        # For each expert, select top-capacity tokens based on routing scores
        for e in range(self.num_experts):
            # Get routing scores and assignment mask for this expert
            expert_scores = router_probs[:, :, e]  # [B, T]
            assign_mask = (expert_indices == e)    # [B, T] - tokens assigned to this expert
            
            # Count assigned tokens
            num_assigned = int(assign_mask.sum().detach().item())
            num_tokens = min(num_assigned, capacity)
            
            if num_tokens > 0:
                # Mask out non-assigned tokens explicitly to avoid ties with zeros
                scores = expert_scores.masked_fill(~assign_mask, float('-inf')).reshape(-1)
                _, top_idx = scores.topk(num_tokens, dim=0)
                
                # Convert to 2D indices
                b_idx = top_idx // T
                t_idx = top_idx % T
                
                # Set dispatch mask
                dispatch_mask[b_idx, t_idx, e] = 1.0
        
        # Compute gates - for Switch with top-1, these become 1.0 for dispatched tokens
        # We can simplify since each token goes to exactly 1 expert or none
        gates = dispatch_mask  # Already 0/1, no need to normalize for Switch
        
        # Update tracking with proper device handling
        if self.training:
            # Ensure all operations are on the same device
            dispatched = dispatch_mask.sum(dim=(0, 1)).detach()
            self.expert_counts.add_(dispatched)
            self.total_tokens.add_(batch_tokens)
            dropped = batch_tokens - int(dispatch_mask.sum().detach().item())
            self.dropped_tokens.add_(dropped)
        
        # Compute auxiliary losses
        aux_loss = self.compute_aux_loss(router_probs, dispatch_mask)
        z_loss = self.compute_z_loss(router_logits)
        
        return gates, dispatch_mask, aux_loss + z_loss
    
    def compute_aux_loss(self, router_probs: torch.Tensor, dispatch_mask: torch.Tensor) -> torch.Tensor:
        """Switch Transformer auxiliary loss for load balancing."""
        # Fraction of tokens dispatched to each expert
        f = dispatch_mask.float().mean(dim=(0, 1))  # [E]
        
        # Probability mass assigned to each expert
        P = router_probs.float().mean(dim=(0, 1))  # [E]
        
        # Auxiliary loss encourages uniform distribution
        aux_loss = self.config.router_aux_loss_weight * (f * P).sum() * self.num_experts
        
        return aux_loss
    
    def compute_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Router z-loss to prevent logit explosion."""
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return self.config.router_z_loss_weight * z_loss
    
    def get_stats(self) -> Dict:
        """Get routing statistics."""
        if self.total_tokens > 0:
            loads = self.expert_counts / self.total_tokens
            drop_rate = float(self.dropped_tokens) / float(self.total_tokens)
            cv = loads.std() / (loads.mean() + 1e-6)
            
            return {
                "expert_loads": loads.tolist(),
                "drop_rate": drop_rate,
                "cv": float(cv),
            }
        return {}


class Expert(nn.Module):
    """Single expert FFN with configurable expansion."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        hidden_dim = int(config.n_embd * config.expert_expansion)
        
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        for module in [self.gate_proj, self.up_proj]:
            nn.init.normal_(module.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02 / math.sqrt(2 * config.n_layer))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class RescueFFN(nn.Module):
    """Small rescue FFN for overflow tokens if overflow_policy='rescue'."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        hidden_dim = int(config.n_embd * config.rescue_expansion)
        
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.n_embd),
        )
        
        # Small initialization for rescue path
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """
    V2 MoE layer with Switch routing and no base MLP.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Router
        self.router = SwitchRouter(config)
        
        # Experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        
        # Optional rescue FFN for overflow
        if config.overflow_policy == "rescue":
            self.rescue_ffn = RescueFFN(config)
        else:
            self.rescue_ffn = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        residual = x
        
        # Get routing decisions
        gates, dispatch_mask, router_loss = self.router(x)
        
        # Flatten for efficient expert processing
        x_flat = x.view(B * T, C)
        output_flat = torch.zeros_like(x_flat)
        
        # Process each expert's batch
        for e, expert in enumerate(self.experts):
            # Find tokens for this expert
            expert_mask = dispatch_mask[:, :, e].view(-1) > 0
            
            if not expert_mask.any():
                continue
            
            # Get indices efficiently
            indices = expert_mask.nonzero(as_tuple=True)[0]
            
            # Batch process through expert
            expert_input = x_flat.index_select(0, indices)
            
            # Pad for FP8/cuBLAS alignment if needed
            num_tokens = expert_input.size(0)
            alignment = 16
            remainder = num_tokens % alignment
            
            if remainder != 0:
                pad_size = alignment - remainder
                padding = torch.zeros(pad_size, C, dtype=expert_input.dtype, device=expert_input.device)
                expert_input_padded = torch.cat([expert_input, padding], dim=0)
                expert_output = expert(expert_input_padded)
                expert_output = expert_output[:num_tokens]  # Remove padding
            else:
                expert_output = expert(expert_input)
            
            # For Switch routing with top-1, gates are 1.0 for dispatched tokens
            # Skip the multiplication for efficiency
            output_flat.index_add_(0, indices, expert_output)
        
        # Reshape output
        output = output_flat.view(B, T, C)
        
        # Handle overflow tokens if using rescue policy
        if self.config.overflow_policy == "rescue" and self.rescue_ffn is not None:
            # Find overflow tokens (not dispatched to any expert)
            overflow_mask = dispatch_mask.sum(dim=-1) == 0  # [B, T]
            
            if overflow_mask.any():
                overflow_input = x[overflow_mask]
                overflow_output = self.rescue_ffn(overflow_input)
                output[overflow_mask] = overflow_output
        
        # Residual connection
        output = residual + output
        
        return output, router_loss


class CausalSelfAttention(nn.Module):
    """Multi-head attention with GQA support and Flash Attention."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Query, Key, Value projections with GQA
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.bias)
        
        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash Attention flag
        self.use_flash = FLASH_AVAILABLE
        
        # Pre-compute causal mask for efficiency
        self._causal_mask = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention
        q = q.view(B, T, self.config.n_head, self.config.head_dim)
        k = k.view(B, T, self.config.n_kv_head, self.config.head_dim)
        v = v.view(B, T, self.config.n_kv_head, self.config.head_dim)
        
        if self.use_flash and self.config.n_head == self.config.n_kv_head:
            # Flash Attention for standard MHA (GQA requires special handling)
            # Flash expects [B, T, H, D] format
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.config.dropout if self.training else 0.0,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.config.head_dim)
            )
            output = output.view(B, T, C)
        else:
            # Fallback: Memory-efficient GQA without repeat_interleave
            q = q.transpose(1, 2)  # [B, H, T, D]
            k = k.transpose(1, 2)  # [B, Hkv, T, D]
            v = v.transpose(1, 2)  # [B, Hkv, T, D]
            
            # Efficient KV repetition using expand (no memory copy)
            repeat_factor = self.config.n_head // self.config.n_kv_head
            k = k[:, :, None, :, :].expand(B, self.config.n_kv_head, repeat_factor, T, self.config.head_dim)
            k = k.reshape(B, self.config.n_head, T, self.config.head_dim)
            v = v[:, :, None, :, :].expand(B, self.config.n_kv_head, repeat_factor, T, self.config.head_dim)
            v = v.reshape(B, self.config.n_head, T, self.config.head_dim)
            
            # Standard scaled dot-product attention
            scale = 1.0 / math.sqrt(self.config.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Causal mask (cache this in __init__ for better performance)
            if not hasattr(self, '_causal_mask') or self._causal_mask.shape[0] < T:
                self._causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            mask = self._causal_mask[:T, :T]
            scores = scores.masked_fill(mask, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        output = self.out_proj(output)
        output = self.resid_dropout(output)
        
        return output


class Block(nn.Module):
    """Transformer block with MoE FFN."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.moe = MoELayer(config)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention
        x = x + self.attn(self.ln1(x))
        
        # MoE FFN
        x, router_loss = self.moe(self.ln2(x))
        
        return x, router_loss


class MoEModelV2(nn.Module):
    """
    V2 MoE model without base MLP, using Switch routing.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm and head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Logging
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        
        print(f"MoE V2 Model initialized:")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.1f}M")
        print(f"  Experts: {config.num_experts}")
        print(f"  Expert expansion: {config.expert_expansion}x")
        print(f"  Capacity factor: {config.capacity_factor}")
        print(f"  Overflow policy: {config.overflow_policy}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple:
        B, T = idx.shape
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        total_router_loss = 0.0
        for block in self.blocks:
            x, router_loss = block(x)
            total_router_loss = total_router_loss + router_loss
        
        # Final layer norm and output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            # Add router loss
            loss = loss + total_router_loss
        
        # Log statistics periodically
        if self.training and self.step % self.config.log_interval == 0:
            self.log_statistics()
        
        self.step.add_(1)
        
        return logits, loss
    
    def log_statistics(self):
        """Log routing statistics for monitoring."""
        print(f"\n[Step {self.step}] Expert Routing Statistics:")
        
        for i, block in enumerate(self.blocks):
            stats = block.moe.router.get_stats()
            if stats:
                loads = stats["expert_loads"]
                drop_rate = stats["drop_rate"]
                cv = stats["cv"]
                
                if i % 6 == 0:  # Log every 6th layer
                    print(f"  Layer {i}: loads={[f'{l:.2f}' for l in loads]}, "
                          f"drop={drop_rate:.1%}, CV={cv:.3f}")
                
                # Reset counters
                block.moe.router.expert_counts.zero_()
                block.moe.router.total_tokens.zero_()
                block.moe.router.dropped_tokens.zero_()
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def estimate_flops_per_token(self) -> int:
        """Estimate FLOPs per token for forward pass."""
        c = self.config
        
        # Attention FLOPs (with GQA)
        attn_flops = c.n_layer * (
            4 * c.n_embd * c.n_embd +  # QKV projections (K,V are smaller with GQA)
            2 * c.n_embd * c.block_size * c.head_dim +  # Attention scores
            2 * c.n_embd * c.n_embd  # Output projection
        )
        
        # MoE FLOPs (only 1 expert active per token with Switch routing)
        expert_dim = int(c.n_embd * c.expert_expansion)
        moe_flops = c.n_layer * (
            c.n_embd * c.num_experts +  # Router
            3 * c.n_embd * expert_dim +  # One expert (gate, up, down)
            c.n_embd  # Residual add
        )
        
        # Embeddings and final projection
        other_flops = 2 * c.vocab_size * c.n_embd
        
        return attn_flops + moe_flops + other_flops


def test_model():
    """Test the V2 MoE model."""
    config = MoEConfig()
    model = MoEModelV2(config).cuda()
    
    # Test input
    batch_size = 4
    seq_len = 512
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    
    # Forward pass
    model.train()
    logits, loss = model(x, x)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Benchmark
    import time
    model.eval()
    torch.cuda.synchronize()
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            model(x)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(20):
            model(x)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
    
    tokens_per_sec = batch_size * seq_len * 20 / elapsed
    ms_per_forward = elapsed / 20 * 1000
    
    print(f"\nPerformance:")
    print(f"  Forward pass: {ms_per_forward:.2f}ms")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    
    # Compute statistics
    params = model.get_num_params()
    flops = model.estimate_flops_per_token()
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {params/1e6:.1f}M")
    print(f"  FLOPs per token: {flops/1e9:.2f}B")
    print(f"  Compute equivalent: ~{flops/1e9 * 6:.0f}M dense model")


if __name__ == "__main__":
    test_model()