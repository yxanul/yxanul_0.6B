#!/usr/bin/env python3
"""Test memory usage with different batch sizes"""

import torch
import sys
sys.path.append('src')

from model_te_v2 import create_te_v2_model, ModelConfig

def test_batch_size(batch_size, seq_len):
    """Test if a batch size fits in memory"""
    print(f"\nTesting batch_size={batch_size}, seq_len={seq_len}")
    
    # Create model
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
    
    # Create dummy batch
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    
    # Print memory before forward
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Memory after model load: {allocated:.2f} GB")
    
    try:
        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, logits = model(input_ids, labels=input_ids)
        
        # Print memory after forward
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"✓ Success!")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  After backward: {allocated:.2f} GB")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"✗ OOM Error!")
        print(f"  {str(e).split('.')[0]}.")
        
    # Clear memory
    del model, input_ids
    if 'loss' in locals():
        del loss
    if 'logits' in locals():
        del logits
    torch.cuda.empty_cache()

def main():
    print("="*60)
    print("Testing Memory Usage on RTX 5090 (32GB)")
    print("="*60)
    
    # Get GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total VRAM: {total_mem:.1f} GB")
    
    # Test different configurations
    configs = [
        (1, 2048),  # Ultra conservative
        (2, 2048),  # Conservative
        (4, 1024),  # Medium sequences
        (8, 512),   # Short sequences
        (16, 256),  # Very short
    ]
    
    for batch_size, seq_len in configs:
        test_batch_size(batch_size, seq_len)
        torch.cuda.synchronize()
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)

if __name__ == "__main__":
    main()