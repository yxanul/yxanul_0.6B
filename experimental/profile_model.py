#!/usr/bin/env python3
"""
Profile the full model to understand performance bottlenecks.
"""
import torch
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import ModelConfig, SimpleGPT

def profile_model():
    """Profile full model forward and backward pass."""
    print("Model Performance Profiling")
    print("=" * 50)
    
    print("Creating model...")
    config = ModelConfig(
        vocab_size=49152,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=4,  # GQA with 3x compression
        block_size=2048,
        dropout=0.0,  # No dropout for testing
    )
    
    model = SimpleGPT(config).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.1f}M")
    
    # Test batch size
    batch_size = 8
    x = torch.randint(0, 49152, (batch_size, 2048)).cuda()
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: 2048")
    print(f"  Tokens per batch: {batch_size * 2048}")
    
    print("\nWarming up...")
    for i in range(5):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, loss = model(x, x)
            loss.backward()
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/5 complete")
    
    # Clear gradients
    model.zero_grad()
    
    print("\nTiming 10 iterations (forward + backward)...")
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(10):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, loss = model(x, x)
            loss.backward()
        torch.cuda.synchronize()
        if i % 2 == 0:
            print(f"  Iteration {i+1}/10")
    
    end = time.time()
    total_time = end - start
    ms_per_iter = total_time / 10 * 1000
    tokens_per_iter = batch_size * 2048
    tok_per_sec = tokens_per_iter / (ms_per_iter / 1000)
    
    print(f"\nResults:")
    print(f"  Time per iteration: {ms_per_iter:.1f}ms")
    print(f"  Tokens/sec (raw): {tok_per_sec:.0f}")
    print(f"  With grad_accum=32: {tok_per_sec*32:.0f} tok/s")
    print(f"  With grad_accum=16: {tok_per_sec*16:.0f} tok/s")
    print(f"  With grad_accum=8: {tok_per_sec*8:.0f} tok/s")
    
    # Test forward only
    print("\nTiming 20 forward passes only...")
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(20):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss = model(x, x)
            torch.cuda.synchronize()
    
    end = time.time()
    forward_time = (end - start) / 20 * 1000
    forward_tok_per_sec = tokens_per_iter / (forward_time / 1000)
    
    print(f"  Forward only: {forward_time:.1f}ms")
    print(f"  Forward tokens/sec: {forward_tok_per_sec:.0f}")
    print(f"  Backward takes: {ms_per_iter - forward_time:.1f}ms ({(ms_per_iter - forward_time)/ms_per_iter*100:.0f}% of time)")

if __name__ == "__main__":
    profile_model()