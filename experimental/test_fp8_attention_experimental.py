#!/usr/bin/env python3
"""
Test script for experimental FP8 attention with corrected shapes.
Tests different backend configurations to isolate issues.
"""

import os
import sys
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# Import the experimental model
from model_te_experimental import ModelConfig, SimpleGPT_TE, get_fp8_recipe

def test_fp8_attention_backends():
    """Test FP8 attention with different backends."""
    print("="*60)
    print("FP8 Attention Backend Testing")
    print("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    try:
        import transformer_engine
        print(f"TransformerEngine Version: {transformer_engine.__version__}")
    except:
        pass
    
    # Test configurations
    backends = [
        ("1", "FlashAttention"),  # Usually works best on consumer GPUs
        ("0", "Math"),            # Fallback, always works but slower
        ("2", "cuDNN"),           # Often fails on consumer GPUs
    ]
    
    # Model configuration
    config = ModelConfig(
        vocab_size=49152,
        n_layer=4,  # Small model for testing
        n_head=12,
        n_embd=768,
        n_kv_heads=3,  # Test GQA
        block_size=512,
        dropout=0.0,  # No dropout for testing
        use_fp8=True
    )
    
    # Test each backend
    results = []
    for backend_id, backend_name in backends:
        print(f"\n{'='*60}")
        print(f"Testing Backend {backend_id}: {backend_name}")
        print("="*60)
        
        # Set backend
        os.environ['NVTE_FUSED_ATTN_BACKEND'] = backend_id
        print(f"Set NVTE_FUSED_ATTN_BACKEND={backend_id}")
        
        try:
            # Create model
            print("Creating model...")
            model = SimpleGPT_TE(config).cuda()
            model = model.to(torch.bfloat16)
            
            # Get FP8 recipe
            fp8_recipe = get_fp8_recipe(config)
            
            # Create test input (divisible by 16 for FP8)
            batch_size = 2
            seq_len = 256  # Divisible by 16
            x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
            
            print(f"Input shape: {x.shape}")
            print("Running forward pass with FP8...")
            
            # Forward with FP8
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                logits, loss = model(x, targets=x)
            
            print(f"✅ Backend {backend_id} ({backend_name}) WORKS!")
            print(f"  Output shape: {logits.shape}")
            print(f"  Loss: {loss.item():.4f}")
            results.append((backend_id, backend_name, "SUCCESS"))
            
            # Test different sequence lengths
            test_lengths = [128, 256, 512, 1024]
            print(f"\nTesting different sequence lengths:")
            for length in test_lengths:
                try:
                    x_test = torch.randint(0, config.vocab_size, (1, length)).cuda()
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        logits_test, _ = model(x_test, targets=x_test)
                    print(f"  Seq length {length}: ✓")
                except Exception as e:
                    print(f"  Seq length {length}: ✗ ({str(e)[:50]}...)")
            
        except Exception as e:
            print(f"❌ Backend {backend_id} ({backend_name}) FAILED!")
            print(f"  Error: {str(e)[:200]}")
            results.append((backend_id, backend_name, f"FAILED: {str(e)[:100]}"))
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for backend_id, backend_name, status in results:
        print(f"Backend {backend_id} ({backend_name:15}): {status}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    successful_backends = [b for b in results if "SUCCESS" in b[2]]
    if successful_backends:
        best_backend = successful_backends[0]
        print(f"✅ Use backend {best_backend[0]} ({best_backend[1]}) for FP8 attention")
        print(f"   Set: export NVTE_FUSED_ATTN_BACKEND={best_backend[0]}")
    else:
        print("❌ No FP8 attention backend works on this GPU")
        print("   FP8 will be used for Linear layers only (still gives speedup)")
    
    return len(successful_backends) > 0

def test_gqa_correctness():
    """Verify GQA expansion is correct."""
    print("\n" + "="*60)
    print("Testing GQA Head Expansion Correctness")
    print("="*60)
    
    B, S, H_q, H_kv, D = 2, 128, 12, 3, 64
    
    # Create dummy tensors
    q = torch.randn(B, S, H_q, D).cuda()
    k = torch.randn(B, S, H_kv, D).cuda()
    v = torch.randn(B, S, H_kv, D).cuda()
    
    print(f"Original shapes:")
    print(f"  Q: {q.shape} (12 heads)")
    print(f"  K: {k.shape} (3 heads)")
    print(f"  V: {v.shape} (3 heads)")
    
    # Expand K/V heads
    repeat_factor = H_q // H_kv
    k_expanded = k.repeat_interleave(repeat_factor, dim=2)
    v_expanded = v.repeat_interleave(repeat_factor, dim=2)
    
    print(f"\nAfter GQA expansion (repeat_interleave on dim=2):")
    print(f"  Q: {q.shape}")
    print(f"  K: {k_expanded.shape}")
    print(f"  V: {v_expanded.shape}")
    
    # Verify correctness
    assert k_expanded.shape == (B, S, H_q, D), "K shape mismatch!"
    assert v_expanded.shape == (B, S, H_q, D), "V shape mismatch!"
    
    # Check that values are repeated correctly
    for i in range(H_kv):
        for j in range(repeat_factor):
            idx = i * repeat_factor + j
            assert torch.allclose(k_expanded[:, :, idx, :], k[:, :, i, :]), f"K repeat error at head {idx}"
            assert torch.allclose(v_expanded[:, :, idx, :], v[:, :, i, :]), f"V repeat error at head {idx}"
    
    print("\n✅ GQA expansion is correct!")

if __name__ == "__main__":
    # Test GQA correctness first
    test_gqa_correctness()
    
    # Test FP8 attention backends
    success = test_fp8_attention_backends()
    
    if success:
        print("\n🎉 FP8 attention is working! You can use the experimental model.")
        print("   Use: python train_fp8.py --model model_te_experimental")
    else:
        print("\n⚠️  FP8 attention not available, but FP8 Linear layers still work!")
        print("   Continue using the standard model_te.py")