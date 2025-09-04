"""
Diagnose which attention path is actually being used and why performance is poor.
"""

import torch
import torch.nn.functional as F
from model_moe_v2 import MoEConfig, MoEModelV2
import time

def test_attention_paths():
    """Test which attention implementation is being used."""
    print("="*60)
    print("Attention Path Diagnosis")
    print("="*60)
    
    # Check Flash Attention availability
    try:
        from flash_attn import flash_attn_func
        print("✓ Flash Attention imported successfully")
        flash_available = True
    except ImportError as e:
        print(f"✗ Flash Attention not available: {e}")
        flash_available = False
    
    # Check PyTorch SDPA availability and backends
    print("\nPyTorch SDPA backends:")
    print(f"  PyTorch version: {torch.__version__}")
    
    # Check which SDPA backends are available
    if hasattr(F, 'scaled_dot_product_attention'):
        print("  ✓ scaled_dot_product_attention available")
        
        # Check backend availability (PyTorch 2.0+)
        if hasattr(torch.backends.cuda, 'sdp_kernel'):
            print("  Checking SDPA kernel preferences:")
            # These attributes exist in PyTorch 2.0+
            backends = ['flash', 'mem_efficient', 'math']
            for backend in backends:
                try:
                    # Check if backend is enabled
                    enabled = getattr(torch.backends.cuda.sdp_kernel, f'enable_{backend}', None)
                    if enabled is not None:
                        print(f"    {backend}: available")
                except:
                    pass
    else:
        print("  ✗ scaled_dot_product_attention NOT available")
    
    # Create a small model to test
    print("\n" + "="*40)
    print("Testing actual attention execution:")
    
    config = MoEConfig(
        n_layer=2,
        n_embd=896,
        n_head=28,
        n_kv_head=7,
        num_experts=4,
        block_size=512,
    )
    
    model = MoEModelV2(config).cuda().bfloat16()
    
    # Check what path the attention module thinks it's using
    attn_module = model.blocks[0].attn
    print(f"\nAttention module state:")
    print(f"  use_flash attribute: {attn_module.use_flash}")
    
    # Test forward pass and see which path executes
    x = torch.randn(2, 256, config.n_embd).cuda().bfloat16()
    
    # Monkey-patch to detect which path is used
    original_forward = attn_module.forward
    path_used = []
    
    def tracked_forward(self, x):
        # Try Flash Attention
        if self.use_flash:
            try:
                from flash_attn import flash_attn_func
                path_used.append("flash")
                print("  → Using Flash Attention path")
            except:
                path_used.append("flash_failed")
                print("  → Flash Attention failed, falling back")
        else:
            path_used.append("no_flash")
        
        # Check if SDPA is used in fallback
        if not self.use_flash or "flash_failed" in path_used:
            try:
                # Test if SDPA works
                test_q = torch.randn(1, 4, 8, 32).cuda().bfloat16()
                test_k = torch.randn(1, 2, 8, 32).cuda().bfloat16()  # GQA
                test_v = torch.randn(1, 2, 8, 32).cuda().bfloat16()
                F.scaled_dot_product_attention(test_q, test_k, test_v, is_causal=True)
                path_used.append("sdpa")
                print("  → Using PyTorch SDPA path")
            except Exception as e:
                path_used.append("manual")
                print(f"  → Using manual attention (slowest): {e}")
        
        return original_forward(x)
    
    attn_module.forward = lambda x: tracked_forward(attn_module, x)
    
    # Run forward
    with torch.no_grad():
        output = attn_module(x)
    
    # Restore original
    attn_module.forward = original_forward
    
    # Benchmark different batch sizes
    print("\n" + "="*40)
    print("Benchmarking attention performance:")
    
    for batch_size in [1, 2, 4, 8]:
        for seq_len in [256, 512, 1024]:
            x = torch.randn(batch_size, seq_len, config.n_embd).cuda().bfloat16()
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = attn_module(x)
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = attn_module(x)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            ms_per_forward = elapsed / 10 * 1000
            print(f"  B={batch_size}, T={seq_len}: {ms_per_forward:.2f}ms")
    
    # Test with explicit SDPA backend selection
    if hasattr(torch.backends.cuda, 'sdp_kernel'):
        print("\n" + "="*40)
        print("Testing with explicit Flash SDPA backend:")
        
        try:
            from torch.backends.cuda import sdp_kernel, SDPBackend
            
            # Force Flash backend
            with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                x = torch.randn(4, 512, config.n_embd).cuda().bfloat16()
                
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    for _ in range(10):
                        _ = attn_module(x)
                
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                print(f"  With Flash backend forced: {elapsed/10*1000:.2f}ms")
        except Exception as e:
            print(f"  Failed to force Flash backend: {e}")
    
    # Memory usage check
    print("\n" + "="*40)
    print("Memory usage analysis:")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(8, 1024, config.n_embd).cuda().bfloat16()
    
    start_mem = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        output = attn_module(x)
    
    peak_mem = torch.cuda.max_memory_allocated()
    
    print(f"  Input size: {x.shape}")
    print(f"  Memory before: {start_mem / 1e6:.1f} MB")
    print(f"  Peak memory: {peak_mem / 1e6:.1f} MB")
    print(f"  Memory increase: {(peak_mem - start_mem) / 1e6:.1f} MB")
    
    # Expected vs actual
    expected_mem = x.numel() * 2  # BF16 = 2 bytes
    actual_mem = peak_mem - start_mem
    ratio = actual_mem / expected_mem if expected_mem > 0 else 0
    
    print(f"  Memory ratio (actual/expected): {ratio:.1f}x")
    
    if ratio > 3:
        print("  ⚠ High memory usage suggests inefficient attention!")

if __name__ == "__main__":
    test_attention_paths()
    
    print("\n" + "="*60)
    print("Diagnosis Summary:")
    print("="*60)
    print("""
    If Flash Attention is not being used:
    1. Install: pip install flash-attn --no-build-isolation
    2. Requires: CUDA 11.6+, SM 7.5+ GPU
    
    If SDPA is slow:
    1. Update PyTorch: pip install torch>=2.0
    2. Use torch.compile() for better fusion
    
    If memory ratio > 3x:
    1. GQA expansion is happening (K,V duplication)
    2. Attention implementation is inefficient
    """)