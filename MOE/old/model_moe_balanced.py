"""
MoE with aggressive load balancing to prevent expert collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass  
class BalancedMoEConfig:
    n_embd: int = 896
    moe_num_experts: int = 3
    
    # Router
    router_type: str = "switch"  # Switch-style routing with capacity
    router_top_k: int = 1
    
    # Load balancing (aggressive)
    load_balance_type: str = "switch"  # Use Switch Transformer's approach
    capacity_factor: float = 1.25  # Allow 25% overflow
    aux_loss_weight: float = 0.1   # Strong balancing pressure
    z_loss_weight: float = 0.001   # Router z-loss
    
    # Expert config
    expert_expansion: float = 4.0  # Larger since no base
    expert_dropout: float = 0.1
    
    # Initialization
    router_init_std: float = 0.02
    expert_init_std: float = 0.02


class SwitchRouter(nn.Module):
    """
    Switch Transformer routing with capacity-based load balancing.
    Ensures balanced expert usage through hard capacity limits.
    """
    
    def __init__(self, config: BalancedMoEConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.n_embd, config.moe_num_experts, bias=False)
        
        # Initialize router with small weights for balanced start
        nn.init.normal_(self.router.weight, std=config.router_init_std)
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        E = self.config.moe_num_experts
        
        # Compute router logits
        router_logits = self.router(x)  # [B, T, E]
        
        # Router z-loss for preventing router saturation
        z_loss = self.compute_z_loss(router_logits)
        
        # Softmax routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-1 expert selection
        expert_index = router_probs.argmax(dim=-1)  # [B, T]
        expert_mask = F.one_hot(expert_index, num_classes=E).float()  # [B, T, E]
        
        # Capacity-based token dropping for load balance
        # Each expert can handle at most capacity_factor * (B*T/E) tokens
        capacity = int(self.config.capacity_factor * B * T / E)
        
        # Count tokens going to each expert
        tokens_per_expert = expert_mask.sum(dim=(0, 1))  # [E]
        
        # Create dispatch mask (which tokens actually get processed)
        dispatch_mask = torch.zeros_like(expert_mask)
        
        for e in range(E):
            # Find tokens going to this expert
            expert_tokens = expert_mask[:, :, e].nonzero(as_tuple=True)
            
            if len(expert_tokens[0]) > 0:
                # Limit to capacity
                num_tokens = min(len(expert_tokens[0]), capacity)
                
                # Random selection if over capacity (or priority-based)
                if len(expert_tokens[0]) > capacity:
                    # Use routing scores as priority
                    scores = router_probs[:, :, e][expert_tokens]
                    _, top_indices = scores.topk(capacity)
                    selected_b = expert_tokens[0][top_indices]
                    selected_t = expert_tokens[1][top_indices]
                else:
                    selected_b = expert_tokens[0][:num_tokens]
                    selected_t = expert_tokens[1][:num_tokens]
                
                dispatch_mask[selected_b, selected_t, e] = 1.0
        
        # Combine weights (only for dispatched tokens)
        gates = router_probs * dispatch_mask
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Auxiliary loss for load balancing
        aux_loss = self.compute_aux_loss(router_probs, dispatch_mask)
        
        return gates, dispatch_mask, aux_loss + z_loss
    
    def compute_aux_loss(self, router_probs, dispatch_mask):
        """Switch Transformer's auxiliary loss."""
        # Fraction of tokens dispatched to each expert
        f = dispatch_mask.float().mean(dim=(0, 1))  # [E]
        
        # Probability mass assigned to each expert
        P = router_probs.float().mean(dim=(0, 1))  # [E]
        
        # Auxiliary loss
        aux_loss = self.config.aux_loss_weight * (f * P).sum() * self.config.moe_num_experts
        
        return aux_loss
    
    def compute_z_loss(self, logits):
        """Router z-loss to prevent logit explosion."""
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return self.config.z_loss_weight * z_loss


class BalancedMoE(nn.Module):
    """
    MoE with Switch-style routing and aggressive load balancing.
    No base MLP for maximum efficiency.
    """
    
    def __init__(self, config: BalancedMoEConfig):
        super().__init__()
        self.config = config
        C = config.n_embd
        E = config.moe_num_experts
        H = int(C * config.expert_expansion)
        
        # Switch router with capacity limits
        self.router = SwitchRouter(config)
        
        # Experts (identical architecture for balance)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C, H, bias=False),
                nn.GELU(),
                nn.Dropout(config.expert_dropout),
                nn.Linear(H, C, bias=False),
            ) for _ in range(E)
        ])
        
        # Initialize experts
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=config.expert_init_std)
        
        # Tracking
        self.register_buffer("expert_counts", torch.zeros(E))
        self.register_buffer("total_tokens", torch.tensor(0, dtype=torch.long))
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        
        # Get routing decisions
        gates, dispatch_mask, router_loss = self.router(x)
        
        # Expert computation
        output = torch.zeros_like(x)
        
        for e, expert in enumerate(self.experts):
            # Find tokens for this expert
            expert_mask = dispatch_mask[:, :, e] > 0
            
            if not expert_mask.any():
                continue
            
            # Process expert batch
            expert_input = x[expert_mask]
            expert_output = expert(expert_input)
            
            # Apply gating weights
            expert_gates = gates[:, :, e][expert_mask].unsqueeze(-1)
            
            # Accumulate output
            output[expert_mask] += expert_gates * expert_output
        
        # Update tracking
        if self.training:
            self.expert_counts += dispatch_mask.sum(dim=(0, 1))
            self.total_tokens += B * T
            
            # Print stats periodically
            if self.total_tokens % 10000 == 0:
                self.print_stats()
        
        # Residual connection (no base MLP!)
        output = x + output
        
        return output, router_loss
    
    def print_stats(self):
        """Print load balancing statistics."""
        if self.total_tokens > 0:
            loads = self.expert_counts / self.total_tokens
            ideal = 1.0 / self.config.moe_num_experts
            
            print(f"\n[Expert Load @ {self.total_tokens} tokens]")
            for i, load in enumerate(loads):
                bar = "█" * int(load * 20)
                print(f"  Expert {i}: {bar:<20} {load:.1%}")
            
            cv = loads.std() / (loads.mean() + 1e-6)
            print(f"  CV: {cv:.3f} (target: <0.3)")
            
            # Reset counters
            self.expert_counts.zero_()
            self.total_tokens.zero_()


class RandomRouter(nn.Module):
    """
    Random uniform routing for testing perfect balance.
    This ensures exactly 1/E tokens go to each expert.
    """
    
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        E = self.num_experts
        
        # Random assignment ensuring balance
        flat_size = B * T
        assignments = torch.randperm(flat_size, device=x.device) % E
        expert_mask = F.one_hot(assignments, num_classes=E).float()
        expert_mask = expert_mask.view(B, T, E)
        
        # Uniform weights
        gates = expert_mask / expert_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        
        return gates, expert_mask, torch.tensor(0.0, device=x.device)


def test_load_balance():
    """Test different routing strategies."""
    import time
    
    config = BalancedMoEConfig()
    
    # Test balanced MoE
    print("Testing Balanced MoE with Switch Router...")
    moe = BalancedMoE(config).cuda()
    
    x = torch.randn(8, 512, config.n_embd).cuda()
    
    # Run some iterations to see load balancing
    moe.train()
    for i in range(20):
        out, loss = moe(x)
        if i % 5 == 0:
            print(f"  Router loss: {loss.item():.4f}")
    
    # Benchmark
    moe.eval()
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            out, _ = moe(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\nPerformance:")
    print(f"  Time: {elapsed/100*1000:.2f}ms")
    print(f"  Throughput: {8*512*100/elapsed:.0f} tokens/sec")
    
    # Test with random router for comparison
    print("\n" + "="*50)
    print("Testing with Random Router (perfect balance)...")
    
    # Replace router with random
    moe.router = RandomRouter(config.moe_num_experts).cuda()
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            out, _ = moe(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed/100*1000:.2f}ms")
    print(f"  Throughput: {8*512*100/elapsed:.0f} tokens/sec")
    print("  (This is the theoretical maximum with perfect balance)")


if __name__ == "__main__":
    test_load_balance()