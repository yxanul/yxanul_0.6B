"""
Optimized MoE without redundant base MLP.
Based on findings: Base MLP adds 2x overhead with no benefit when k=1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class OptimizedMoEConfig:
    """Configuration for optimized MoE."""
    n_embd: int = 896
    moe_num_experts: int = 3
    
    # Router configuration
    router_type: str = "softmax_topk"  # Better for k=1, ensures exactly 1 expert
    router_top_k: int = 1               # Always activate exactly k experts
    router_noise_std: float = 0.01      # Add noise for exploration during training
    router_z_loss_coeff: float = 0.01   # Router z-loss for load balancing
    
    # Expert configuration  
    expert_expansion: float = 3.5       # Larger experts since no base MLP
    expert_dropout: float = 0.1
    
    # Load balancing
    load_balancing_type: str = "aux_loss"  # or "z_loss" or "switch"
    aux_loss_coeff: float = 0.01
    
    # Monitoring
    track_expert_metrics: bool = True
    

class RouterMetrics:
    """Track router statistics for monitoring."""
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.reset()
    
    def reset(self):
        self.expert_counts = torch.zeros(self.num_experts)
        self.expert_scores = []
        self.load_balance_losses = []
        self.total_tokens = 0
    
    def update(self, routing_weights, active_mask):
        """Update metrics with batch routing info."""
        # Count tokens per expert
        batch_counts = active_mask.sum(dim=(0, 1))  # [E]
        self.expert_counts += batch_counts.cpu()
        self.total_tokens += active_mask.shape[0] * active_mask.shape[1]
        
        # Track routing scores
        self.expert_scores.append(routing_weights.mean(dim=(0, 1)).cpu())
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        if self.total_tokens == 0:
            return {}
        
        loads = self.expert_counts / self.total_tokens
        ideal_load = 1.0 / self.num_experts
        load_imbalance = (loads - ideal_load).abs().mean()
        
        return {
            "expert_loads": loads.tolist(),
            "load_imbalance": float(load_imbalance),
            "most_used_expert": int(self.expert_counts.argmax()),
            "least_used_expert": int(self.expert_counts.argmin()),
            "cv": float(loads.std() / (loads.mean() + 1e-6)),  # Coefficient of variation
        }


class OptimizedMoE(nn.Module):
    """
    Optimized MoE layer without base MLP overhead.
    - Softmax top-k routing for exactly k experts
    - Auxiliary loss for load balancing
    - Expert metrics tracking
    """
    
    def __init__(self, config: OptimizedMoEConfig):
        super().__init__()
        self.config = config
        C = config.n_embd
        E = config.moe_num_experts
        H = int(C * config.expert_expansion)
        
        # Router (smaller, more focused)
        self.router = nn.Sequential(
            nn.Linear(C, C // 2),
            nn.ReLU(),
            nn.Linear(C // 2, E)
        )
        
        # Experts (larger since no base MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C, H),
                nn.ReLU(),
                nn.Dropout(config.expert_dropout),
                nn.Linear(H, C)
            ) for _ in range(E)
        ])
        
        # Metrics tracking
        if config.track_expert_metrics:
            self.metrics = RouterMetrics(E)
        else:
            self.metrics = None
    
    def compute_aux_loss(self, routing_probs, active_mask):
        """Compute auxiliary loss for load balancing."""
        # Fraction of tokens going to each expert
        f = active_mask.float().mean(dim=(0, 1))  # [E]
        
        # Probability mass going to each expert  
        P = routing_probs.float().mean(dim=(0, 1))  # [E]
        
        # Auxiliary loss encourages uniform distribution
        aux_loss = self.config.aux_loss_coeff * (f * P).sum() * self.config.moe_num_experts
        
        return aux_loss
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        
        # Compute routing logits
        routing_logits = self.router(x)  # [B, T, E]
        
        # Add noise during training for exploration
        if self.training and self.config.router_noise_std > 0:
            noise = torch.randn_like(routing_logits) * self.config.router_noise_std
            routing_logits = routing_logits + noise
        
        # Softmax routing
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Top-k selection
        k = self.config.router_top_k
        topk_vals, topk_idxs = torch.topk(routing_probs, k, dim=-1)
        
        # Create active mask
        active_mask = torch.zeros_like(routing_probs)
        active_mask.scatter_(-1, topk_idxs, 1.0)
        
        # Renormalize weights
        routing_weights = routing_probs * active_mask
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Update metrics
        if self.metrics is not None:
            self.metrics.update(routing_weights, active_mask)
        
        # Expert computation (optimized)
        output = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            # Get mask for this expert
            expert_mask = active_mask[:, :, e] > 0
            
            if not expert_mask.any():
                continue
            
            # Process tokens for this expert
            expert_input = x[expert_mask]
            expert_output = expert(expert_input)
            expert_weights = routing_weights[:, :, e][expert_mask]
            
            # Weighted add back
            output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output
        
        # Compute auxiliary loss
        aux_loss = None
        if self.training:
            aux_loss = self.compute_aux_loss(routing_probs, active_mask)
        
        # Residual connection (no base MLP!)
        output = x + output
        
        return output, aux_loss
    

class MonitoredMoE(OptimizedMoE):
    """MoE with detailed monitoring capabilities."""
    
    def __init__(self, config: OptimizedMoEConfig):
        super().__init__(config)
        self.register_buffer("expert_visit_counts", torch.zeros(config.moe_num_experts))
        self.register_buffer("total_routed", torch.tensor(0, dtype=torch.long))
        
    def forward(self, x: torch.Tensor):
        output, aux_loss = super().forward(x)
        
        # Log every N steps
        if self.training and self.total_routed % 1000 == 0:
            stats = self.metrics.get_stats()
            print(f"\n[MoE Stats @ {self.total_routed} tokens]")
            print(f"  Expert loads: {[f'{l:.3f}' for l in stats['expert_loads']]}")
            print(f"  Load imbalance: {stats['load_imbalance']:.3f}")
            print(f"  CV: {stats['cv']:.3f}")
            self.metrics.reset()
        
        self.total_routed += x.shape[0] * x.shape[1]
        
        return output, aux_loss


def test_optimized_moe():
    """Test the optimized MoE performance."""
    config = OptimizedMoEConfig()
    moe = MonitoredMoE(config).cuda()
    
    # Test forward pass
    x = torch.randn(8, 512, config.n_embd).cuda()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            out, _ = moe(x)
    
    # Benchmark
    torch.cuda.synchronize()
    import time
    start = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            out, _ = moe(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\nOptimized MoE Performance:")
    print(f"  Time per forward: {elapsed/100*1000:.2f}ms")
    print(f"  Throughput: {8*512*100/elapsed:.0f} tokens/sec")
    
    # Show metrics
    if moe.metrics:
        stats = moe.metrics.get_stats()
        print(f"\nRouting Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    test_optimized_moe()