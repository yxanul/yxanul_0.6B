#!/usr/bin/env python3
"""
Test raw attention performance to diagnose A100 bottleneck.
"""
import torch
import torch.nn.functional as F
import time

def test_attention_speed():
    """Test raw attention performance and backend detection."""
    device = 'cuda'
    batch = 8
    heads = 12
    seq_len = 2048
    head_dim = 64
    
    print("Testing Attention Performance on A100")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Batch size: {batch}")
    print(f"  Heads: {heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Total shape: [{batch}, {heads}, {seq_len}, {head_dim}]")
    print()
    
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
    
    # Time default backend
    print("Timing default attention backend...")
    start = time.time()
    for _ in range(100):
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
    end = time.time()
    
    ms_per_forward = (end-start)/100*1000
    print(f"\nDefault attention: {ms_per_forward:.2f}ms per forward")
    
    # Calculate theoretical tokens/sec
    tokens_per_batch = batch * seq_len
    theoretical_tok_per_sec = (tokens_per_batch / (ms_per_forward/1000))
    print(f"Theoretical max: {theoretical_tok_per_sec:.0f} tokens/sec for attention alone")
    
    # Check backends
    print(f"\nBackends enabled:")
    print(f"  Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  Mem efficient SDP: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  Math SDP: {torch.backends.cuda.math_sdp_enabled()}")
    
    # Test math-only backend for comparison
    print("\nTesting math-only backend for comparison...")
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_mem_efficient=False,
        enable_math=True
    ):
        start = time.time()
        for _ in range(100):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
        end = time.time()
        math_time = (end-start)/100*1000
    
    print(f"Math-only attention: {math_time:.2f}ms")
    if math_time > ms_per_forward:
        print(f"GOOD: Flash/Efficient attention IS being used!")
        print(f"Speedup from Flash: {math_time/ms_per_forward:.1f}x")
    else:
        print(f"WARNING: Flash attention may NOT be active!")
        print(f"Performance ratio: {math_time/ms_per_forward:.2f}")
    
    # Test with Flash explicitly enabled (if available)
    try:
        print("\nTrying to force Flash Attention...")
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=False,
            enable_math=False
        ):
            start = time.time()
            for _ in range(100):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()
            end = time.time()
            flash_time = (end-start)/100*1000
            print(f"Flash-only attention: {flash_time:.2f}ms")
    except Exception as e:
        print(f"Could not force Flash attention: {e}")

if __name__ == "__main__":
    test_attention_speed()