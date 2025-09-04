"""Test MTP overhead to confirm it's the bottleneck."""

import torch
import time
from model_moe import ModelConfig, OptimizedGPT_GLMini_PRMoE

def test_mtp_overhead():
    # Config with and without MTP
    config_with_mtp = ModelConfig()
    config_without_mtp = ModelConfig()
    config_without_mtp.use_mtp = False
    
    # Create models
    model_with_mtp = OptimizedGPT_GLMini_PRMoE(config_with_mtp).cuda().bfloat16()
    model_without_mtp = OptimizedGPT_GLMini_PRMoE(config_without_mtp).cuda().bfloat16()
    
    # Test input
    batch_size = 6
    seq_len = 2048
    x = torch.randint(0, config_with_mtp.vocab_size, (batch_size, seq_len)).cuda()
    targets = x.clone()
    
    # Warmup
    print("Warmup...")
    for _ in range(3):
        with torch.no_grad():
            _, _ = model_with_mtp(x, targets)
            _, _ = model_without_mtp(x, targets)
    
    # Test WITH MTP
    print("\n=== WITH Multi-Token Prediction ===")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            logits, loss = model_with_mtp(x, targets)
    torch.cuda.synchronize()
    time_with_mtp = (time.time() - start) / 10 * 1000
    print(f"Average forward pass WITH MTP: {time_with_mtp:.2f}ms")
    
    # Test WITHOUT MTP
    print("\n=== WITHOUT Multi-Token Prediction ===")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            logits, loss = model_without_mtp(x, targets)
    torch.cuda.synchronize()
    time_without_mtp = (time.time() - start) / 10 * 1000
    print(f"Average forward pass WITHOUT MTP: {time_without_mtp:.2f}ms")
    
    print(f"\n=== OVERHEAD ===")
    print(f"MTP adds: {time_with_mtp - time_without_mtp:.2f}ms")
    print(f"Slowdown factor: {time_with_mtp / time_without_mtp:.2f}x")
    
    # Break down MTP cost
    print(f"\n=== MTP BREAKDOWN ===")
    print(f"MTP offsets: {config_with_mtp.mtp_offsets}")
    print(f"MTP weights: {config_with_mtp.mtp_weights}")
    print(f"Number of extra forward passes: {len(config_with_mtp.mtp_offsets)}")
    print(f"Expected overhead per pass: {(time_with_mtp - time_without_mtp) / len(config_with_mtp.mtp_offsets):.2f}ms")

if __name__ == "__main__":
    test_mtp_overhead()