#!/usr/bin/env python3
"""
Test if torch.compile improves performance on A100.
"""
import torch
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import ModelConfig, SimpleGPT

def test_compile():
    """Compare performance with and without torch.compile."""
    print("Torch Compile Performance Test")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    config = ModelConfig(
        vocab_size=49152,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=4,
        block_size=2048,
        dropout=0.0,
    )
    
    batch_size = 8
    x = torch.randint(0, 49152, (batch_size, 2048)).cuda()
    
    # Test without compile
    print("Testing WITHOUT torch.compile...")
    model_normal = SimpleGPT(config).cuda()
    
    # Warmup
    print("  Warming up...")
    for _ in range(10):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, loss = model_normal(x, x)
            loss.backward()
        model_normal.zero_grad()
        torch.cuda.synchronize()
    
    # Time it
    print("  Timing 10 iterations...")
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(10):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, loss = model_normal(x, x)
            loss.backward()
        model_normal.zero_grad()
        torch.cuda.synchronize()
    
    end = time.time()
    normal_time = (end - start) / 10 * 1000
    normal_tok_sec = (batch_size * 2048) / (normal_time / 1000)
    
    print(f"  Time per iter: {normal_time:.1f}ms")
    print(f"  Tokens/sec: {normal_tok_sec:.0f}")
    
    # Test with compile
    print("\nTesting WITH torch.compile...")
    model_compiled = SimpleGPT(config).cuda()
    
    try:
        print("  Compiling model (this takes ~30 seconds)...")
        model_compiled = torch.compile(model_compiled, mode="default")
        
        # Warmup (triggers compilation)
        print("  Warming up (first iteration compiles)...")
        for i in range(10):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss = model_compiled(x, x)
                loss.backward()
            model_compiled.zero_grad()
            torch.cuda.synchronize()
            if i == 0:
                print("    Compilation complete, continuing warmup...")
        
        # Time it
        print("  Timing 10 iterations...")
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(10):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss = model_compiled(x, x)
                loss.backward()
            model_compiled.zero_grad()
            torch.cuda.synchronize()
        
        end = time.time()
        compiled_time = (end - start) / 10 * 1000
        compiled_tok_sec = (batch_size * 2048) / (compiled_time / 1000)
        
        print(f"  Time per iter: {compiled_time:.1f}ms")
        print(f"  Tokens/sec: {compiled_tok_sec:.0f}")
        
        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Normal model: {normal_time:.1f}ms ({normal_tok_sec:.0f} tok/s)")
        print(f"Compiled model: {compiled_time:.1f}ms ({compiled_tok_sec:.0f} tok/s)")
        print(f"Speedup: {normal_time/compiled_time:.2f}x")
        print(f"\nWith grad_accum=32:")
        print(f"  Normal: {normal_tok_sec*32:.0f} tok/s")
        print(f"  Compiled: {compiled_tok_sec*32:.0f} tok/s")
        
    except Exception as e:
        print(f"  ERROR: Could not compile model: {e}")
        print("  This might be due to PyTorch version or CUDA compatibility")

if __name__ == "__main__":
    test_compile()