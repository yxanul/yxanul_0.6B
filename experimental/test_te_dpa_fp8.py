#!/usr/bin/env python3
"""
Minimal test for TransformerEngine DotProductAttention with FP8.
Tests if DPA works at all with FP8 on RTX 5090.
"""

import os
# Set larger workspace for GEMM operations
os.environ.setdefault("NVTE_GEMM_WORKSPACE_SIZE", str(128*1024*1024))
os.environ.setdefault("NVTE_FUSED_ATTN_BACKEND", "1")  # 1=FlashAttention

import torch
import torch.nn.functional as F

try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import DotProductAttention
    from transformer_engine.common.recipe import DelayedScaling, Format
except ImportError:
    print("ERROR: TransformerEngine not found!")
    exit(1)

def test_dpa_fp8_minimal():
    """Test DPA with FP8 using minimal configuration."""
    print("="*60)
    print("Minimal DPA FP8 Test")
    print("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"Backend: {os.environ.get('NVTE_FUSED_ATTN_BACKEND', 'auto')}")
    print(f"Workspace: {os.environ.get('NVTE_GEMM_WORKSPACE_SIZE', 'default')}")
    
    # Test parameters
    B, S, Hq, Hkv, D = 4, 2048, 12, 2, 64  # GQA: n_head=12, n_kv_heads=2
    print(f"\nTest config: B={B}, S={S}, Hq={Hq}, Hkv={Hkv}, D={D}")
    print(f"GQA ratio: {Hq//Hkv}x compression")
    
    try:
        # Create FP8 recipe
        print("\nCreating FP8 recipe...")
        recipe = DelayedScaling(fp8_format=Format.HYBRID, fp8_dpa=True)
        
        # Create DPA module
        print("Creating DPA module...")
        dpa = DotProductAttention(
            num_attention_heads=Hq,
            kv_channels=D,
            attn_mask_type="causal",
            attention_dropout=0.1
        )
        
        # Create test tensors
        print("Creating test tensors...")
        xq = torch.randn(B, S, Hq, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        xk = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        xv = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        
        print(f"Original shapes - Q: {xq.shape}, K: {xk.shape}, V: {xv.shape}")
        
        # Expand K/V for GQA
        rep = Hq // Hkv
        print(f"Expanding K/V heads by {rep}x...")
        xk = xk.repeat_interleave(rep, dim=2).contiguous()
        xv = xv.repeat_interleave(rep, dim=2).contiguous()
        
        print(f"After GQA - Q: {xq.shape}, K: {xk.shape}, V: {xv.shape}")
        
        # Run with FP8
        print("\nRunning DPA with FP8...")
        with te.fp8_autocast(True, fp8_recipe=recipe):
            y = dpa(xq.contiguous(), xk, xv)  # [B, S, Hq, D]
            print(f"Output shape: {y.shape}")
            
            # Compute loss and backward
            loss = (y.float()**2).mean()
            
        print("Running backward pass...")
        loss.backward()
        
        print(f"\n✅ TE DPA FP8 WORKS! Loss: {float(loss):.4f}")
        return True
        
    except Exception as e:
        print(f"\n❌ TE DPA FP8 FAILED!")
        print(f"Error: {str(e)[:200]}")
        return False

def test_different_configs():
    """Test various configurations to find what works."""
    print("\n" + "="*60)
    print("Testing Different Configurations")
    print("="*60)
    
    configs = [
        # (B, S, Hq, Hkv, D, name)
        (2, 256, 12, 3, 64, "Small (our model)"),
        (4, 512, 12, 3, 64, "Medium"),
        (4, 1024, 12, 3, 64, "Large"),
        (4, 2048, 12, 3, 64, "XLarge"),
        (4, 2048, 12, 2, 64, "XLarge 6x GQA"),
        (4, 2048, 16, 4, 64, "XLarge 4x GQA 16H"),
    ]
    
    results = []
    for B, S, Hq, Hkv, D, name in configs:
        print(f"\nTesting {name}: B={B}, S={S}, Hq={Hq}, Hkv={Hkv}, D={D}")
        
        try:
            recipe = DelayedScaling(fp8_format=Format.HYBRID, fp8_dpa=True)
            dpa = DotProductAttention(
                num_attention_heads=Hq,
                kv_channels=D,
                attn_mask_type="causal",
                attention_dropout=0.0
            )
            
            xq = torch.randn(B, S, Hq, D, device='cuda', dtype=torch.bfloat16)
            xk = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16)
            xv = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16)
            
            # Expand for GQA
            rep = Hq // Hkv
            xk = xk.repeat_interleave(rep, dim=2).contiguous()
            xv = xv.repeat_interleave(rep, dim=2).contiguous()
            
            with te.fp8_autocast(True, fp8_recipe=recipe):
                y = dpa(xq, xk, xv)
            
            print(f"  ✓ {name} works!")
            results.append((name, "SUCCESS"))
            
        except Exception as e:
            print(f"  ✗ {name} failed: {str(e)[:50]}...")
            results.append((name, f"FAILED: {str(e)[:30]}"))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, status in results:
        print(f"{name:20}: {status}")

def test_backends():
    """Test different attention backends."""
    print("\n" + "="*60)
    print("Testing Different Backends")
    print("="*60)
    
    backends = [
        ("0", "Math"),
        ("1", "FlashAttention"),
        ("2", "cuDNN"),
    ]
    
    B, S, Hq, Hkv, D = 2, 256, 12, 3, 64
    
    for backend_id, backend_name in backends:
        print(f"\nTesting backend {backend_id} ({backend_name})...")
        os.environ["NVTE_FUSED_ATTN_BACKEND"] = backend_id
        
        try:
            recipe = DelayedScaling(fp8_format=Format.HYBRID, fp8_dpa=True)
            dpa = DotProductAttention(
                num_attention_heads=Hq,
                kv_channels=D,
                attn_mask_type="causal",
                attention_dropout=0.0
            )
            
            xq = torch.randn(B, S, Hq, D, device='cuda', dtype=torch.bfloat16)
            xk = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16)
            xv = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16)
            
            # Expand for GQA
            rep = Hq // Hkv
            xk = xk.repeat_interleave(rep, dim=2).contiguous()
            xv = xv.repeat_interleave(rep, dim=2).contiguous()
            
            with te.fp8_autocast(True, fp8_recipe=recipe):
                y = dpa(xq, xk, xv)
            
            print(f"  ✅ Backend {backend_id} ({backend_name}) WORKS!")
            
        except Exception as e:
            print(f"  ❌ Backend {backend_id} ({backend_name}) FAILED: {str(e)[:100]}")

if __name__ == "__main__":
    # Run minimal test
    success = test_dpa_fp8_minimal()
    
    if success:
        # If minimal test works, try other configs
        test_different_configs()
        test_backends()
    else:
        print("\nBasic DPA FP8 test failed. This GPU may not support FP8 attention.")
        print("FP8 will still work for Linear layers (as shown in your training).")