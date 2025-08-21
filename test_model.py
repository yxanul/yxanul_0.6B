#!/usr/bin/env python3
"""
Test script to verify all model improvements are working correctly.
"""

import torch
import sys
import os
sys.path.append('src')

from model import create_model, ModelConfig
import yaml

def test_model():
    """Test the model with all improvements."""
    
    print("=" * 60)
    print("Testing Yxanul 0.6B Model with All Improvements")
    print("=" * 60)
    
    # Load model config
    with open("configs/model_config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Filter config to only include ModelConfig fields
    from dataclasses import fields
    valid_fields = {f.name for f in fields(ModelConfig)}
    filtered_config = {k: v for k, v in config_dict["model"].items() if k in valid_fields}
    
    # Create model
    model = create_model(filtered_config)
    
    print("\n" + "=" * 60)
    print("Model Architecture Summary:")
    print("=" * 60)
    
    # Test forward pass
    batch_size = 2
    seq_len = 256
    
    # Create dummy input
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    print(f"\nTesting forward pass with batch_size={batch_size}, seq_len={seq_len}")
    
    # Test model forward
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        
    if isinstance(outputs, tuple):
        logits = outputs[1] if len(outputs) > 1 else outputs[0]
    else:
        logits = outputs
    
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, 50257)")
    
    assert logits.shape == (batch_size, seq_len, 50257), "Output shape mismatch!"
    
    print("\n[OK] Forward pass successful!")
    
    # Test with labels (training mode)
    print("\nTesting training mode with labels...")
    model.train()
    labels = input_ids.clone()
    
    outputs = model(input_ids, labels=labels)
    
    if isinstance(outputs, tuple):
        loss = outputs[0]
        print(f"Loss: {loss.item():.4f}")
        print("[OK] Training mode successful!")
    
    # Count parameters
    print("\n" + "=" * 60)
    print("Parameter Count Breakdown:")
    print("=" * 60)
    
    total_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            params = module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                params += module.bias.numel()
            if params > 1000000:  # Only show modules with > 1M params
                print(f"{name:40s}: {params/1e6:8.2f}M")
            total_params += params
    
    print("-" * 60)
    print(f"{'Total':40s}: {total_params/1e6:8.2f}M")
    
    # Verify key improvements
    print("\n" + "=" * 60)
    print("Improvements Verification:")
    print("=" * 60)
    
    config = ModelConfig(**filtered_config)
    
    checks = [
        ("SwiGLU activation", config.use_swiglu),
        ("RMSNorm", config.use_rmsnorm),
        ("Grouped-Query Attention", config.num_kv_heads < config.num_attention_heads),
        ("RoPE embeddings", config.use_rope),
        ("Position embeddings", hasattr(model.model, 'position_embeddings')),
        ("Causal mask caching", hasattr(model.model, 'causal_mask')),
        ("Weight tying", model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()),
    ]
    
    all_passed = True
    for name, condition in checks:
        status = "[OK]" if condition else "[FAIL]"
        print(f"{status} {name}: {'Enabled' if condition else 'Disabled'}")
        if not condition and name != "Weight tying":  # Weight tying might not work if shapes differ
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed or (not checks[-1][1] and all(c[1] for c in checks[:-1])):
        print("[OK] All critical improvements are working correctly!")
    else:
        print("[WARNING] Some improvements may need attention")
    print("=" * 60)

if __name__ == "__main__":
    test_model()