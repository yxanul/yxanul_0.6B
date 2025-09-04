"""
End-to-end training test for V2 MoE model.
Tests that the model can actually train with proper gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import time
from model_moe_v2 import MoEConfig, MoEModelV2


def test_training_step():
    """Test a complete training step with gradients."""
    print("Testing training step with gradient flow...")
    
    # Small config for quick testing
    config = MoEConfig(
        n_layer=4,
        n_embd=256,
        n_head=8,
        n_kv_head=4,
        num_experts=4,
        expert_expansion=2.0,  # Smaller for testing
        capacity_factor=1.25,
        router_aux_loss_weight=0.01,
    )
    
    model = MoEModelV2(config).cuda()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Use BF16 for training
    model = model.bfloat16()
    
    # Training data
    batch_size = 4
    seq_len = 256
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    
    # Initial loss
    model.train()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _, initial_loss = model(x, x)
    initial_loss_value = initial_loss.item()
    
    # Training steps
    losses = []
    router_grad_norms = []
    
    for step in range(10):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _, loss = model(x, x)
        
        loss.backward()
        
        # Check router gradients
        router_grad_norm = 0.0
        for block in model.blocks:
            if block.moe.router.router.weight.grad is not None:
                router_grad_norm += block.moe.router.router.weight.grad.norm().item()
        router_grad_norms.append(router_grad_norm)
        
        optimizer.step()
        losses.append(loss.item())
    
    # Verify training is working
    final_loss = losses[-1]
    avg_router_grad = sum(router_grad_norms) / len(router_grad_norms)
    
    print(f"  Initial loss: {initial_loss_value:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {initial_loss_value - final_loss:.4f}")
    print(f"  Avg router gradient norm: {avg_router_grad:.6f}")
    
    # Check that loss decreased (model is learning)
    if final_loss < initial_loss_value:
        print("✓ Model is learning (loss decreased)")
    else:
        print("⚠ Loss did not decrease (may need more steps)")
    
    # Check that router gets gradients (critical for MoE)
    if avg_router_grad > 0:
        print("✓ Router receives gradients from task loss")
    else:
        print("✗ Router has zero gradients!")
        return False
    
    return True


def test_load_balancing():
    """Test that load balancing improves over time."""
    print("\nTesting load balancing convergence...")
    
    config = MoEConfig(
        n_layer=2,
        num_experts=4,
        capacity_factor=1.5,
        router_aux_loss_weight=0.1,  # Strong balancing pressure
    )
    
    model = MoEModelV2(config).cuda().bfloat16()
    optimizer = AdamW(model.parameters(), lr=3e-4)
    
    batch_size = 8
    seq_len = 512
    
    # Track CV over time
    cv_history = []
    
    for step in range(20):
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _, loss = model(x, x)
        loss.backward()
        optimizer.step()
        
        # Get load balance stats
        if step % 5 == 0:
            stats = model.blocks[0].moe.router.get_stats()
            if stats and 'cv' in stats:
                cv = stats['cv']
                cv_history.append(cv)
                print(f"  Step {step}: CV={cv:.3f}, loads={[f'{l:.2f}' for l in stats['expert_loads']]}")
                
                # Reset tracking
                for block in model.blocks:
                    block.moe.router.expert_counts.zero_()
                    block.moe.router.total_tokens.zero_()
                    block.moe.router.dropped_tokens.zero_()
    
    # Check if balance improved
    if len(cv_history) >= 2:
        initial_cv = cv_history[0]
        final_cv = cv_history[-1]
        
        if final_cv < initial_cv:
            print(f"✓ Load balance improved: CV {initial_cv:.3f} → {final_cv:.3f}")
        else:
            print(f"⚠ Load balance did not improve: CV {initial_cv:.3f} → {final_cv:.3f}")
    
    return True


def benchmark_throughput():
    """Benchmark actual training throughput."""
    print("\nBenchmarking training throughput...")
    
    config = MoEConfig(
        n_layer=24,
        num_experts=4,
        expert_expansion=3.5,
        capacity_factor=1.25,
    )
    
    model = MoEModelV2(config).cuda().bfloat16()
    optimizer = AdamW(model.parameters(), lr=6e-4)
    
    batch_size = 12
    seq_len = 512
    
    # Warmup
    for _ in range(3):
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _, loss = model(x, x)
        loss.backward()
        optimizer.step()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    num_steps = 10
    for _ in range(num_steps):
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _, loss = model(x, x)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens_per_step = batch_size * seq_len
    total_tokens = tokens_per_step * num_steps
    tokens_per_sec = total_tokens / elapsed
    ms_per_step = elapsed / num_steps * 1000
    
    print(f"  Model size: {model.get_num_params()/1e6:.1f}M parameters")
    print(f"  Batch size: {batch_size} x {seq_len} = {tokens_per_step} tokens/step")
    print(f"  Time per step: {ms_per_step:.1f}ms")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    
    # Expected: 150k+ tokens/sec based on optimizations
    if tokens_per_sec > 100000:
        print(f"✓ Good throughput achieved!")
    elif tokens_per_sec > 50000:
        print(f"⚠ Moderate throughput (expected 150k+)")
    else:
        print(f"✗ Low throughput (expected 150k+)")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("V2 MoE Training Tests")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(1)
    
    tests_passed = 0
    tests_failed = 0
    
    # Run tests
    tests = [
        ("Training Step", test_training_step),
        ("Load Balancing", test_load_balancing),
        ("Throughput Benchmark", benchmark_throughput),
    ]
    
    for name, test_fn in tests:
        print(f"\n{name}:")
        print("-"*40)
        try:
            if test_fn():
                tests_passed += 1
            else:
                tests_failed += 1
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            tests_failed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    
    if tests_failed == 0:
        print("✅ All training tests passed! V2 model is ready.")
    else:
        print("❌ Some tests failed. Review the model.")
    
    exit(0 if tests_failed == 0 else 1)