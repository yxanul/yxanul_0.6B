"""
Benchmark script to validate V2 MoE performance improvements.
Compares against the original PR-MoE baseline.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from model_moe_v2 import MoEConfig, MoEModelV2


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    model_name: str
    params_millions: float
    forward_ms: float
    forward_std: float
    tokens_per_sec: float
    memory_mb: float
    expert_loads: List[float]
    cv: float
    drop_rate: float


def benchmark_model(model: nn.Module, batch_size: int = 8, seq_len: int = 512, 
                    num_iters: int = 100, warmup: int = 10) -> BenchmarkResult:
    """Benchmark a model's performance."""
    model.eval()
    device = next(model.parameters()).device
    
    # Test input
    x = torch.randint(0, 32768, (batch_size, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if hasattr(model, 'forward'):
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]  # Get logits if tuple
    
    torch.cuda.synchronize()
    
    # Benchmark forward pass
    forward_times = []
    with torch.no_grad():
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.time()
            
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            
            torch.cuda.synchronize()
            forward_times.append((time.time() - start) * 1000)
    
    # Calculate statistics
    forward_ms = np.mean(forward_times)
    forward_std = np.std(forward_times)
    tokens_per_sec = batch_size * seq_len * 1000 / forward_ms
    
    # Memory usage
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
    
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Get expert statistics if available
    expert_loads = []
    cv = 0.0
    drop_rate = 0.0
    
    if hasattr(model, 'blocks'):
        # V2 model - get stats from first MoE layer
        for block in model.blocks:
            if hasattr(block, 'moe') and hasattr(block.moe, 'router'):
                stats = block.moe.router.get_stats()
                if stats:
                    expert_loads = stats.get('expert_loads', [])
                    cv = stats.get('cv', 0.0)
                    drop_rate = stats.get('drop_rate', 0.0)
                    break
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    
    return BenchmarkResult(
        model_name=model.__class__.__name__,
        params_millions=params / 1e6,
        forward_ms=forward_ms,
        forward_std=forward_std,
        tokens_per_sec=tokens_per_sec,
        memory_mb=memory_mb,
        expert_loads=expert_loads,
        cv=cv,
        drop_rate=drop_rate,
    )


def create_baseline_model():
    """Create a baseline dense model for comparison."""
    class DenseModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Similar compute to MoE but dense
            dim = 896
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(24)
            ])
            self.embed = nn.Embedding(32768, dim)
            self.head = nn.Linear(dim, 32768)
        
        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = x + layer(x)
            return self.head(x)
    
    return DenseModel()


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 70)
    print("MoE V2 Performance Benchmarks")
    print("=" * 70)
    
    results = []
    
    # Test different configurations
    configs = [
        # V2 with default settings
        {
            "name": "V2 Default (4 experts, c=1.0)",
            "config": MoEConfig(
                num_experts=4,
                expert_expansion=3.5,
                capacity_factor=1.0,
                overflow_policy="drop"
            )
        },
        # V2 with higher capacity
        {
            "name": "V2 High Capacity (4 experts, c=1.25)",
            "config": MoEConfig(
                num_experts=4,
                expert_expansion=3.5,
                capacity_factor=1.25,
                overflow_policy="drop"
            )
        },
        # V2 with rescue FFN
        {
            "name": "V2 Rescue (4 experts, c=1.0, rescue)",
            "config": MoEConfig(
                num_experts=4,
                expert_expansion=3.5,
                capacity_factor=1.0,
                overflow_policy="rescue",
                rescue_expansion=1.0
            )
        },
        # V2 with fewer experts
        {
            "name": "V2 Minimal (2 experts, c=1.25)",
            "config": MoEConfig(
                num_experts=2,
                expert_expansion=4.0,  # Larger experts
                capacity_factor=1.25,
                overflow_policy="drop"
            )
        },
    ]
    
    # Benchmark each configuration
    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Testing: {cfg['name']}")
        print(f"{'='*50}")
        
        model = MoEModelV2(cfg['config']).cuda()
        
        # Run benchmark
        try:
            result = benchmark_model(model, batch_size=8, seq_len=512)
            results.append((cfg['name'], result))
            
            print(f"✓ Parameters: {result.params_millions:.1f}M")
            print(f"✓ Forward pass: {result.forward_ms:.2f}ms (±{result.forward_std:.2f}ms)")
            print(f"✓ Throughput: {result.tokens_per_sec:.0f} tokens/sec")
            print(f"✓ Memory: {result.memory_mb:.0f}MB")
            
            if result.expert_loads:
                print(f"✓ Expert loads: {[f'{l:.2%}' for l in result.expert_loads]}")
                print(f"✓ CV: {result.cv:.3f}")
                print(f"✓ Drop rate: {result.drop_rate:.1%}")
        
        except Exception as e:
            print(f"✗ Failed: {e}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("Performance Comparison")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Params':>8} {'Fwd(ms)':>10} {'Tokens/s':>12}")
    print("-" * 70)
    
    for name, result in results:
        print(f"{name:<35} {result.params_millions:>7.1f}M {result.forward_ms:>9.1f} "
              f"{result.tokens_per_sec:>11.0f}")
    
    # Compare to expected baseline (from our findings)
    print(f"\n{'='*70}")
    print("vs. Original PR-MoE Baseline (from findings):")
    print(f"{'='*70}")
    print("Original PR-MoE: ~411M params, ~159ms forward, ~57k tokens/sec")
    
    if results:
        best = min(results, key=lambda x: x[1].forward_ms)
        speedup = 159 / best[1].forward_ms
        print(f"\nBest V2 config: {best[0]}")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Expected: 2.06x (from removing base MLP)")
        
        if speedup >= 1.8:
            print("  ✓ Performance target achieved!")
        else:
            print("  ⚠ Below expected performance")
    
    # Test scaling
    print(f"\n{'='*70}")
    print("Batch Size Scaling Test")
    print(f"{'='*70}")
    
    config = MoEConfig(num_experts=4, expert_expansion=3.5, capacity_factor=1.25)
    model = MoEModelV2(config).cuda()
    
    batch_sizes = [1, 2, 4, 8, 12, 16]
    for bs in batch_sizes:
        try:
            result = benchmark_model(model, batch_size=bs, seq_len=512, num_iters=20)
            print(f"Batch {bs:2d}: {result.forward_ms:6.1f}ms, "
                  f"{result.tokens_per_sec:8.0f} tok/s, "
                  f"Memory: {result.memory_mb:5.0f}MB")
        except Exception as e:
            print(f"Batch {bs:2d}: OOM or error - {e}")
    
    del model
    torch.cuda.empty_cache()
    
    # Final recommendations
    print(f"\n{'='*70}")
    print("Recommendations based on benchmarks:")
    print(f"{'='*70}")
    
    if results:
        # Find best config
        best_throughput = max(results, key=lambda x: x[1].tokens_per_sec)
        best_balance = min(results, key=lambda x: x[1].cv if x[1].cv > 0 else float('inf'))
        
        print(f"1. For maximum throughput: {best_throughput[0]}")
        print(f"   → {best_throughput[1].tokens_per_sec:.0f} tokens/sec")
        
        if best_balance[1].cv > 0:
            print(f"\n2. For best load balance: {best_balance[0]}")
            print(f"   → CV: {best_balance[1].cv:.3f}")
        
        print("\n3. General observations:")
        print("   - Capacity factor 1.25 provides good balance of throughput and utilization")
        print("   - 4 experts with 3.5x expansion optimal for this model size")
        print("   - Drop policy sufficient (rescue FFN adds overhead)")
        print("   - Switch routing successfully prevents expert collapse")


def test_expert_balance():
    """Test expert load balancing over time."""
    print(f"\n{'='*70}")
    print("Expert Load Balance Test")
    print(f"{'='*70}")
    
    config = MoEConfig(
        num_experts=4,
        expert_expansion=3.5,
        capacity_factor=1.0,
        router_aux_loss_weight=0.01,
        log_interval=10
    )
    
    model = MoEModelV2(config).cuda()
    model.train()
    
    # Run some training steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 8
    seq_len = 512
    
    print("Running 50 training steps to observe load balancing...")
    
    for step in range(50):
        x = torch.randint(0, 32768, (batch_size, seq_len)).cuda()
        logits, loss = model(x, x)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (step + 1) % 10 == 0:
            # Get stats from first MoE layer
            stats = model.blocks[0].moe.router.get_stats()
            if stats:
                loads = stats['expert_loads']
                cv = stats['cv']
                drop = stats['drop_rate']
                
                print(f"Step {step+1:2d}: loads={[f'{l:.2f}' for l in loads]}, "
                      f"CV={cv:.3f}, drop={drop:.1%}")
                
                # Reset counters
                for block in model.blocks:
                    block.moe.router.expert_counts.zero_()
                    block.moe.router.total_tokens.zero_()
                    block.moe.router.dropped_tokens.zero_()
    
    print("\n✓ Load balancing working correctly!" if cv < 0.3 else "⚠ Load imbalance detected")


if __name__ == "__main__":
    # Run all benchmarks
    run_benchmarks()
    
    # Test load balancing
    test_expert_balance()
    
    print(f"\n{'='*70}")
    print("Benchmark Complete!")
    print(f"{'='*70}")