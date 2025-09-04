"""
Test Flash Attention's actual GQA support and find the right configuration.
"""

import torch
import time

def test_flash_attention_gqa():
    """Test if Flash Attention actually supports our GQA configuration."""
    print("="*60)
    print("Testing Flash Attention GQA Support")
    print("="*60)
    
    try:
        from flash_attn import flash_attn_func
        from flash_attn import __version__ as flash_version
        print(f"Flash Attention version: {flash_version}")
    except ImportError:
        print("Flash Attention not installed!")
        return
    
    # Test configurations
    B, T = 2, 512
    n_head = 28
    n_kv_head = 7
    head_dim = 32
    
    print(f"\nTesting with:")
    print(f"  Query heads: {n_head}")
    print(f"  KV heads: {n_kv_head}")
    print(f"  Head dim: {head_dim}")
    print(f"  Batch: {B}, Seq: {T}")
    
    # Create tensors in BF16
    q = torch.randn(B, T, n_head, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, T, n_kv_head, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, T, n_kv_head, head_dim, device='cuda', dtype=torch.bfloat16)
    
    print("\nTest 1: Direct Flash Attention call with GQA")
    try:
        # Flash Attention v2 should handle GQA directly
        output = flash_attn_func(
            q, k, v,
            causal=True,
            softmax_scale=1.0 / (head_dim ** 0.5)
        )
        print(f"  ✓ Success! Output shape: {output.shape}")
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            output = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  Performance: {elapsed/100*1000:.2f}ms per forward")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Flash Attention may not support GQA with this configuration")
    
    print("\nTest 2: Flash Attention with KV repetition (fallback)")
    try:
        # Manually repeat KV to match Q heads
        repeat_factor = n_head // n_kv_head
        k_repeated = k.repeat_interleave(repeat_factor, dim=2)
        v_repeated = v.repeat_interleave(repeat_factor, dim=2)
        
        print(f"  K shape after repeat: {k_repeated.shape}")
        print(f"  V shape after repeat: {v_repeated.shape}")
        
        output = flash_attn_func(
            q, k_repeated, v_repeated,
            causal=True,
            softmax_scale=1.0 / (head_dim ** 0.5)
        )
        print(f"  ✓ Success with repetition! Output shape: {output.shape}")
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            k_repeated = k.repeat_interleave(repeat_factor, dim=2)
            v_repeated = v.repeat_interleave(repeat_factor, dim=2)
            output = flash_attn_func(q, k_repeated, v_repeated, causal=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  Performance (with repeat): {elapsed/100*1000:.2f}ms per forward")
        
    except Exception as e:
        print(f"  ✗ Failed even with repetition: {e}")
    
    print("\nTest 3: Check Flash Attention's window_size parameter for GQA")
    try:
        # Some versions of Flash Attention use window_size for GQA
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_func
        print("  Additional Flash Attention functions available:")
        print("    - flash_attn_qkvpacked_func")
        print("    - flash_attn_varlen_func")
        
    except ImportError:
        print("  Only basic flash_attn_func available")
    
    print("\nTest 4: PyTorch SDPA with GQA")
    import torch.nn.functional as F
    
    # Transpose for SDPA format [B, H, T, D]
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    
    try:
        output = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            is_causal=True,
            scale=1.0 / (head_dim ** 0.5)
        )
        print(f"  ✓ SDPA works with GQA! Output shape: {output.shape}")
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            output = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                is_causal=True
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  Performance: {elapsed/100*1000:.2f}ms per forward")
        
    except Exception as e:
        print(f"  ✗ SDPA failed with GQA: {e}")
    
    print("\nTest 5: Memory usage comparison")
    torch.cuda.empty_cache()
    
    # Flash with GQA (if it works)
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    
    try:
        for _ in range(10):
            output = flash_attn_func(q, k, v, causal=True)
        flash_mem = torch.cuda.max_memory_allocated() - start_mem
        print(f"  Flash (GQA): {flash_mem/1e6:.1f} MB")
    except:
        flash_mem = None
        print("  Flash (GQA): Failed")
    
    torch.cuda.empty_cache()
    
    # Flash with repetition
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    
    try:
        for _ in range(10):
            k_repeated = k.repeat_interleave(repeat_factor, dim=2)
            v_repeated = v.repeat_interleave(repeat_factor, dim=2)
            output = flash_attn_func(q, k_repeated, v_repeated, causal=True)
        repeat_mem = torch.cuda.max_memory_allocated() - start_mem
        print(f"  Flash (repeated): {repeat_mem/1e6:.1f} MB")
    except:
        repeat_mem = None
        print("  Flash (repeated): Failed")
    
    torch.cuda.empty_cache()
    
    # SDPA
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    
    for _ in range(10):
        output = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            is_causal=True
        )
    sdpa_mem = torch.cuda.max_memory_allocated() - start_mem
    print(f"  SDPA: {sdpa_mem/1e6:.1f} MB")
    
    # Analysis
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    
    if flash_mem is not None and repeat_mem is not None:
        if flash_mem < repeat_mem:
            print("✓ Use Flash Attention with native GQA")
        else:
            print("✗ Flash Attention GQA is not efficient")
    
    if sdpa_mem < (repeat_mem or float('inf')):
        print("✓ PyTorch SDPA is most efficient for GQA")
    
    print("""
If Flash Attention doesn't support GQA natively:
1. Use PyTorch's scaled_dot_product_attention (it handles GQA well)
2. Ensure you're using PyTorch 2.0+ for best SDPA performance
3. DON'T repeat K,V tensors - it causes 4x memory usage!
""")

if __name__ == "__main__":
    test_flash_attention_gqa()