#!/usr/bin/env python3
"""
Minimal test of TransformerEngine FP8 on RTX 5090.
Based on official TE documentation example.
Tests if FP8 works at all on consumer Blackwell GPUs.
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling, MXFP8BlockScaling

def test_basic_fp8():
    """Test basic FP8 functionality with a single Linear layer."""
    print("="*60)
    print("TransformerEngine FP8 Minimal Test")
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
    
    print("\n" + "="*60)
    print("Test 1: Standard DelayedScaling (H100-style)")
    print("="*60)
    
    try:
        # Standard FP8 recipe (H100-style)
        fp8_format = Format.HYBRID  # E4M3 forward, E5M2 backward
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
        
        # Create a simple linear layer
        torch.manual_seed(12345)
        my_linear = te.Linear(768, 768, bias=True).cuda()
        
        # Create input (both dims divisible by 16)
        inp = torch.rand((1024, 768)).cuda()
        
        print("Running forward pass with standard FP8...")
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out_fp8 = my_linear(inp)
        
        print("✓ Forward pass successful!")
        
        # Test backward (outside autocast as per docs)
        print("Running backward pass...")
        loss_fp8 = out_fp8.mean()
        loss_fp8.backward()
        print("✓ Backward pass successful!")
        print("\n✅ Standard FP8 WORKS on this GPU!")
        
    except Exception as e:
        print(f"❌ Standard FP8 failed: {e}")
        
    print("\n" + "="*60)
    print("Test 2: MXFP8BlockScaling (Blackwell-optimized)")
    print("="*60)
    
    try:
        # Check if MXFP8 is available
        try:
            # MXFP8 recipe (Blackwell-style)
            mxfp8_format = Format.E4M3  # E4M3 everywhere
            mxfp8_recipe = MXFP8BlockScaling(fp8_format=mxfp8_format)
            print("MXFP8BlockScaling is available!")
        except:
            print("MXFP8BlockScaling not available in this TE version")
            return False
        
        # Create a new linear layer
        torch.manual_seed(12345)
        my_linear_mx = te.Linear(768, 768, bias=True).cuda()
        
        # Same input
        inp = torch.rand((1024, 768)).cuda()
        
        print("Running forward pass with MXFP8...")
        with te.fp8_autocast(enabled=True, fp8_recipe=mxfp8_recipe):
            out_mxfp8 = my_linear_mx(inp)
        
        print("✓ Forward pass successful!")
        
        # Test backward
        print("Running backward pass...")
        loss_mxfp8 = out_mxfp8.mean()
        loss_mxfp8.backward()
        print("✓ Backward pass successful!")
        print("\n✅ MXFP8 WORKS on this GPU!")
        
    except Exception as e:
        print(f"❌ MXFP8 failed: {e}")
    
    print("\n" + "="*60)
    print("Test 3: Small Transformer Block")
    print("="*60)
    
    try:
        # Try with a small transformer block
        class SimpleTransformerLayer(torch.nn.Module):
            def __init__(self, hidden_size=768, num_attention_heads=12):
                super().__init__()
                self.ln1 = te.LayerNorm(hidden_size)
                self.self_attention = te.MultiheadAttention(
                    hidden_size,
                    num_attention_heads,
                )
                self.ln2 = te.LayerNorm(hidden_size)
                self.mlp = te.Sequential(
                    te.Linear(hidden_size, hidden_size * 4),
                    te.GELU(),
                    te.Linear(hidden_size * 4, hidden_size),
                )
            
            def forward(self, x):
                # Self-attention block
                attn_input = self.ln1(x)
                attn_output = self.self_attention(attn_input, attn_input, attn_input)
                x = x + attn_output
                
                # MLP block
                mlp_input = self.ln2(x)
                mlp_output = self.mlp(mlp_input)
                x = x + mlp_output
                
                return x
        
        print("Creating transformer layer...")
        model = SimpleTransformerLayer().cuda()
        
        # Input with seq_len divisible by 16
        batch_size = 2
        seq_len = 128  # Divisible by 16
        hidden_size = 768
        inp = torch.rand((batch_size, seq_len, hidden_size)).cuda()
        
        # Try both recipes
        for name, recipe in [("Standard", fp8_recipe), ("MXFP8", mxfp8_recipe)]:
            try:
                print(f"\nTesting {name} recipe with transformer...")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                    out = model(inp)
                
                loss = out.mean()
                print(f"  Forward pass: ✓")
                
                optimizer.zero_grad()
                loss.backward()
                print(f"  Backward pass: ✓")
                
                optimizer.step()
                print(f"  Optimizer step: ✓")
                print(f"✅ {name} works with transformer!")
                
            except Exception as e:
                print(f"❌ {name} failed with transformer: {e}")
                
    except Exception as e:
        print(f"❌ Transformer test setup failed: {e}")
    
    print("\n" + "="*60)
    print("Test 4: Different Tensor Sizes")
    print("="*60)
    
    # Test different sizes to find what works
    sizes_to_test = [
        (16, 16),      # Minimum
        (256, 768),    # Small batch
        (512, 768),    # Medium batch
        (1024, 768),   # Large batch
        (2048, 768),   # Very large
    ]
    
    my_linear = te.Linear(768, 768).cuda()
    
    for batch_size, hidden_size in sizes_to_test:
        try:
            inp = torch.rand((batch_size, hidden_size)).cuda()
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = my_linear(inp)
            out.mean().backward()
            print(f"✓ Size ({batch_size}, {hidden_size}) works")
        except Exception as e:
            print(f"✗ Size ({batch_size}, {hidden_size}) failed: {str(e)[:50]}...")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = test_basic_fp8()
    
    if success:
        print("\nFP8 appears to be working! The issue might be in our specific implementation.")
    else:
        print("\nFP8 has fundamental issues on this GPU. Consider using BF16 instead.")
        print("RTX 5090 BF16 performance (164k tok/s) is already excellent!")