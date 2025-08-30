#!/usr/bin/env python3
"""
Test if GQA (Grouped Query Attention) is causing the performance bottleneck.
"""
import torch
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import ModelConfig, SimpleGPT

def test_gqa_performance():
    """Compare performance with and without GQA."""
    print("GQA Performance Comparison")
    print("=" * 50)
    
    # Test configurations
    configs = [
        ("Normal MHA (no GQA)", {
            "vocab_size": 49152,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "n_kv_heads": 12,  # Same as n_head = no GQA
            "block_size": 2048,
            "dropout": 0.0,
        }),
        ("GQA 3x (4 KV heads)", {
            "vocab_size": 49152,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "n_kv_heads": 4,  # GQA with 3x compression
            "block_size": 2048,
            "dropout": 0.0,
        }),
        ("GQA 2x (6 KV heads)", {
            "vocab_size": 49152,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "n_kv_heads": 6,  # GQA with 2x compression
            "block_size": 2048,
            "dropout": 0.0,
        }),
    ]
    
    batch_size = 8
    x = torch.randint(0, 49152, (batch_size, 2048)).cuda()
    
    results = []
    
    for name, config_dict in configs:
        print(f"\nTesting: {name}")
        print(f"  KV heads: {config_dict['n_kv_heads']}")
        print(f"  Q heads: {config_dict['n_head']}")
        print(f"  Compression: {config_dict['n_head']/config_dict['n_kv_heads']:.1f}x")
        
        config = ModelConfig(**config_dict)
        model = SimpleGPT(config).cuda()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params/1e6:.1f}M")
        
        # Warmup
        print("  Warming up...")
        for _ in range(5):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss = model(x, x)
                loss.backward()
            torch.cuda.synchronize()
        
        # Clear gradients
        model.zero_grad()
        
        # Time forward pass
        print("  Timing forward pass...")
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            for _ in range(20):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits, loss = model(x, x)
                torch.cuda.synchronize()
        
        end = time.time()
        forward_time = (end - start) / 20 * 1000
        
        # Time forward + backward
        print("  Timing forward + backward...")
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(10):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss = model(x, x)
                loss.backward()
            torch.cuda.synchronize()
            model.zero_grad()
        
        end = time.time()
        full_time = (end - start) / 10 * 1000
        
        tokens_per_batch = batch_size * 2048
        forward_tok_sec = tokens_per_batch / (forward_time / 1000)
        full_tok_sec = tokens_per_batch / (full_time / 1000)
        
        results.append({
            "name": name,
            "forward_ms": forward_time,
            "full_ms": full_time,
            "forward_tok_sec": forward_tok_sec,
            "full_tok_sec": full_tok_sec,
        })
        
        print(f"  Forward: {forward_time:.1f}ms ({forward_tok_sec:.0f} tok/s)")
        print(f"  Full: {full_time:.1f}ms ({full_tok_sec:.0f} tok/s)")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    baseline = results[0]["full_tok_sec"]
    for r in results:
        speedup = r["full_tok_sec"] / baseline
        print(f"\n{r['name']}:")
        print(f"  Forward: {r['forward_ms']:.1f}ms")
        print(f"  Full: {r['full_ms']:.1f}ms")
        print(f"  Tokens/sec: {r['full_tok_sec']:.0f}")
        print(f"  vs baseline: {speedup:.2f}x")
        print(f"  With grad_accum=32: {r['full_tok_sec']*32:.0f} tok/s")

if __name__ == "__main__":
    test_gqa_performance()