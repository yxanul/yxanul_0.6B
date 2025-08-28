#!/usr/bin/env python3
"""
Test TransformerEngine compatibility and FP8 support.
Diagnose cuBLAS issues with different configurations.
"""

import torch
import sys

def test_basic_cuda():
    """Test basic CUDA functionality."""
    print("="*60)
    print("CUDA Environment Check")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
        
        # Memory info
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU memory: {mem_total:.1f} GB total")
    print()

def test_transformer_engine():
    """Test TransformerEngine installation and features."""
    print("="*60)
    print("TransformerEngine Check")
    print("="*60)
    
    try:
        import transformer_engine as te
        import transformer_engine.pytorch as tep
        from transformer_engine.common import recipe
        
        print(f"TransformerEngine version: {te.__version__}")
        print("TransformerEngine imported successfully!")
        
        # Check FP8 support
        print("\nFP8 Support:")
        print(f"  FP8 available: True")
        print(f"  Formats: E4M3, E5M2, HYBRID")
        
        return True
        
    except ImportError as e:
        print(f"TransformerEngine not available: {e}")
        return False
    except Exception as e:
        print(f"TransformerEngine error: {e}")
        return False

def test_te_linear_sizes():
    """Test different tensor sizes with TE Linear layers."""
    print("\n" + "="*60)
    print("Testing TE Linear Layer Dimensions")
    print("="*60)
    
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        
        # Test dimensions
        test_configs = [
            # (batch_size, seq_len, in_features, out_features)
            (1, 128, 768, 768),      # Standard attention
            (1, 128, 768, 2048),     # FFN up projection  
            (1, 128, 2048, 768),     # FFN down projection
            (1, 128, 768, 50264),    # LM head (vocab)
            (64, 128, 768, 768),     # Full batch
            (64, 128, 768, 2304),    # QKV projection (768*3)
        ]
        
        fp8_recipe = recipe.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=recipe.Format.E4M3,
        )
        
        for batch, seq, in_feat, out_feat in test_configs:
            try:
                # Create layer
                layer = te.Linear(in_feat, out_feat, bias=False).cuda()
                
                # Create input
                x = torch.randn(batch, seq, in_feat).cuda()
                
                # Test forward pass
                with te.fp8_autocast(enabled=True, calibrating=True, fp8_recipe=fp8_recipe):
                    y = layer(x)
                
                # Test backward pass
                loss = y.mean()
                loss.backward()
                
                print(f"✓ [{batch:2}, {seq:3}, {in_feat:5}] -> {out_feat:5} : Success")
                
            except Exception as e:
                error_msg = str(e)
                if "cuBLAS" in error_msg:
                    print(f"✗ [{batch:2}, {seq:3}, {in_feat:5}] -> {out_feat:5} : cuBLAS error")
                elif "divisible" in error_msg:
                    print(f"✗ [{batch:2}, {seq:3}, {in_feat:5}] -> {out_feat:5} : Alignment error")
                else:
                    print(f"✗ [{batch:2}, {seq:3}, {in_feat:5}] -> {out_feat:5} : {error_msg[:50]}")
        
        print("\nNote: Failures indicate dimension incompatibility with FP8/cuBLAS")
        
    except ImportError:
        print("TransformerEngine not available for testing")
    except Exception as e:
        print(f"Testing error: {e}")

def test_workaround_dimensions():
    """Test workaround dimensions that might work better."""
    print("\n" + "="*60)
    print("Testing Alternative Dimensions")
    print("="*60)
    
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        
        # Alternative dimensions (all divisible by 16)
        alt_configs = [
            # More aggressive padding
            (64, 128, 768, 768),     # Keep these
            (64, 128, 768, 2048),    # FFN already aligned
            (64, 128, 768, 50304),   # Vocab padded to 50304 (divisible by 64)
            (64, 128, 768, 2304),    # QKV (768*3)
        ]
        
        fp8_recipe = recipe.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=recipe.Format.E4M3,
        )
        
        print("Testing with more aggressive alignment:")
        for batch, seq, in_feat, out_feat in alt_configs:
            try:
                layer = te.Linear(in_feat, out_feat, bias=False).cuda()
                x = torch.randn(batch, seq, in_feat).cuda()
                
                with te.fp8_autocast(enabled=True, calibrating=True, fp8_recipe=fp8_recipe):
                    y = layer(x)
                
                loss = y.mean()
                loss.backward()
                
                print(f"✓ [{batch:2}, {seq:3}, {in_feat:5}] -> {out_feat:5} : Works!")
                
            except Exception as e:
                print(f"✗ [{batch:2}, {seq:3}, {in_feat:5}] -> {out_feat:5} : Failed")
    
    except ImportError:
        print("TransformerEngine not available")

def test_native_fp8():
    """Test native PyTorch FP8 if available (for newer versions)."""
    print("\n" + "="*60)
    print("Native PyTorch FP8 Support")
    print("="*60)
    
    # Check for torch._C._cuda_getFP8Properties (PyTorch 2.1+)
    if hasattr(torch._C, '_cuda_getFP8Properties'):
        print("PyTorch has native FP8 support!")
        props = torch._C._cuda_getFP8Properties()
        print(f"FP8 properties: {props}")
    else:
        print("PyTorch version does not have native FP8 support")
        print("TransformerEngine is required for FP8 operations")

def main():
    """Run all diagnostics."""
    print("\nTransformerEngine FP8 Compatibility Diagnostics")
    print("="*60 + "\n")
    
    # Basic checks
    test_basic_cuda()
    
    # TE checks
    has_te = test_transformer_engine()
    
    if has_te:
        # Detailed TE tests
        test_te_linear_sizes()
        test_workaround_dimensions()
    
    # Native FP8 check
    test_native_fp8()
    
    print("\n" + "="*60)
    print("Diagnostics Complete")
    print("="*60)
    
    if has_te:
        print("\nRecommendations:")
        print("1. If cuBLAS errors occur, try padding dimensions to multiples of 64")
        print("2. Consider using HYBRID format instead of E4M3")
        print("3. Update TransformerEngine to latest version")
        print("4. Ensure CUDA/cuDNN versions match TE requirements")
    else:
        print("\nPlease install TransformerEngine for FP8 support:")
        print("  pip install transformer-engine")

if __name__ == "__main__":
    main()