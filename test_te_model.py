#!/usr/bin/env python3
"""Test script to verify TE v2.4 model works correctly"""

import torch
import sys
sys.path.append('src')

try:
    import transformer_engine as te
    import transformer_engine.pytorch as te_pytorch
    from transformer_engine.common.recipe import DelayedScaling, Format
    print(f"TransformerEngine v{te.__version__} available")
    TE_AVAILABLE = True
except ImportError:
    print("TransformerEngine not available - this test requires NGC container")
    sys.exit(1)

from model_te_v2 import create_te_v2_model, ModelConfig

def test_model_creation():
    """Test that model can be created without fp8_model_init"""
    print("\n1. Testing model creation...")
    
    config = ModelConfig(
        vocab_size=200005,
        hidden_size=896,
        intermediate_size=2400,
        num_hidden_layers=32,
        num_attention_heads=14,
        num_kv_heads=2,
        use_fp8=True,
        use_factorized_embedding=True,
        factorization_dim=128
    )
    
    model = create_te_v2_model(config)
    print("✓ Model created successfully without fp8_model_init")
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params/1e6:.1f}M")
    
    return model

def test_forward_pass(model):
    """Test forward pass with fp8_autocast"""
    print("\n2. Testing forward pass with FP8...")
    
    batch_size = 2
    seq_len = 128
    
    # Create dummy input
    input_ids = torch.randint(0, 200005, (batch_size, seq_len)).cuda()
    
    # Create FP8 recipe
    fp8_recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,
        amax_compute_algo="max",
        reduce_amax=True
    )
    
    # Forward pass with fp8_autocast
    with te_pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        loss, logits = model(input_ids, labels=input_ids)
    
    print(f"✓ Forward pass successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits shape: {logits.shape}")
    
    return loss

def test_backward_pass(model, loss):
    """Test that backward pass works outside fp8_autocast"""
    print("\n3. Testing backward pass (outside fp8_autocast)...")
    
    # Backward pass MUST be outside fp8_autocast
    loss.backward()
    
    # Check gradients exist
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"✓ Backward pass successful")
    print(f"  Gradient norm: {grad_norm:.4f}")

def main():
    print("="*60)
    print("Testing TE v2.4 Model without fp8_model_init")
    print("="*60)
    
    # Test 1: Create model
    model = test_model_creation()
    
    # Test 2: Forward pass with FP8
    loss = test_forward_pass(model)
    
    # Test 3: Backward pass
    test_backward_pass(model, loss)
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    main()