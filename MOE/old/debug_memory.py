"""
Debug memory usage of V2 model to find OOM cause.
"""

import torch
import gc
from model_moe_v2 import MoEConfig, MoEModelV2

def format_bytes(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def get_memory_stats():
    """Get current GPU memory stats."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return {
            'allocated': format_bytes(allocated),
            'reserved': format_bytes(reserved),
            'allocated_raw': allocated,
            'reserved_raw': reserved,
        }
    return {}

def test_model_creation():
    """Test memory usage during model creation."""
    print("="*60)
    print("Testing Model Creation Memory Usage")
    print("="*60)
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("\nInitial memory:")
    print(get_memory_stats())
    
    # Create config
    config = MoEConfig(
        n_layer=24,
        n_embd=896,
        n_head=28,
        n_kv_head=7,
        num_experts=4,
        expert_expansion=3.5,
        capacity_factor=1.25,
    )
    
    print("\nCreating model...")
    model = MoEModelV2(config).cuda()
    
    print("\nAfter model creation:")
    stats = get_memory_stats()
    print(stats)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    param_memory = total_params * 4  # FP32 = 4 bytes per param
    print(f"\nModel parameters: {total_params/1e6:.1f}M")
    print(f"Expected param memory: {format_bytes(param_memory)}")
    print(f"Actual allocated: {stats['allocated']}")
    
    # Test forward pass with tiny batch
    print("\n" + "="*40)
    print("Testing forward pass with batch_size=1, seq_len=128")
    
    x = torch.randint(0, config.vocab_size, (1, 128)).cuda()
    print(f"Input memory: {format_bytes(x.element_size() * x.numel())}")
    
    print("\nBefore forward:")
    print(get_memory_stats())
    
    with torch.no_grad():
        logits, loss = model(x, x)
    
    print("\nAfter forward:")
    print(get_memory_stats())
    
    # Check for memory leaks in specific components
    print("\n" + "="*40)
    print("Checking component memory usage...")
    
    # Check attention
    print("\nTesting attention module alone...")
    gc.collect()
    torch.cuda.empty_cache()
    
    attn_input = torch.randn(1, 128, config.n_embd).cuda()
    print(f"Before attention: {get_memory_stats()['allocated']}")
    
    with torch.no_grad():
        attn_out = model.blocks[0].attn(attn_input)
    
    print(f"After attention: {get_memory_stats()['allocated']}")
    
    # Check MoE
    print("\nTesting MoE layer alone...")
    gc.collect()
    torch.cuda.empty_cache()
    
    moe_input = torch.randn(1, 128, config.n_embd).cuda()
    print(f"Before MoE: {get_memory_stats()['allocated']}")
    
    with torch.no_grad():
        moe_out, _ = model.blocks[0].moe(moe_input)
    
    print(f"After MoE: {get_memory_stats()['allocated']}")
    
    # Test with different sequence lengths
    print("\n" + "="*40)
    print("Testing memory scaling with sequence length...")
    
    for seq_len in [128, 256, 512, 1024, 2048]:
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            x = torch.randint(0, config.vocab_size, (1, seq_len)).cuda()
            
            before = get_memory_stats()['allocated_raw']
            
            with torch.no_grad():
                logits, _ = model(x, x)
            
            after = get_memory_stats()['allocated_raw']
            
            print(f"Seq {seq_len:4d}: {format_bytes(after - before)} used")
            
        except torch.cuda.OutOfMemoryError:
            print(f"Seq {seq_len:4d}: OOM!")
            break
    
    # Check if it's the causal mask
    print("\n" + "="*40)
    print("Checking causal mask size...")
    
    for block in model.blocks:
        if hasattr(block.attn, '_causal_mask') and block.attn._causal_mask is not None:
            mask_size = block.attn._causal_mask.element_size() * block.attn._causal_mask.numel()
            print(f"Causal mask shape: {block.attn._causal_mask.shape}")
            print(f"Causal mask memory: {format_bytes(mask_size)}")
            break

def test_minimal_config():
    """Test with minimal config to isolate issue."""
    print("\n" + "="*60)
    print("Testing Minimal Configuration")
    print("="*60)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Very small model
    config = MoEConfig(
        n_layer=2,  # Only 2 layers
        n_embd=256,  # Small embedding
        n_head=8,
        n_kv_head=4,
        num_experts=2,  # Fewer experts
        expert_expansion=2.0,  # Smaller expansion
        block_size=512,  # Smaller context
        capacity_factor=1.0,
    )
    
    print(f"Creating minimal model...")
    model = MoEModelV2(config).cuda()
    
    stats = get_memory_stats()
    print(f"Model created: {stats['allocated']}")
    
    # Test forward
    x = torch.randint(0, config.vocab_size, (1, 128)).cuda()
    
    try:
        with torch.no_grad():
            logits, _ = model(x, x)
        print("✓ Minimal model works!")
        
        # Now try with bfloat16
        model = model.bfloat16()
        logits, _ = model(x, x)
        print("✓ BF16 minimal model works!")
        
    except torch.cuda.OutOfMemoryError:
        print("✗ Even minimal model OOMs!")

if __name__ == "__main__":
    # Get GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {format_bytes(total_memory)}")
    
    try:
        test_model_creation()
    except Exception as e:
        print(f"\n✗ Error during model test: {e}")
    
    try:
        test_minimal_config()
    except Exception as e:
        print(f"\n✗ Error during minimal test: {e}")
    
    # Final memory state
    print("\n" + "="*60)
    print("Final Memory State:")
    print(get_memory_stats())
    
    # Suggestions
    print("\nPossible causes of OOM:")
    print("1. GQA expansion in attention (4x memory for K/V)")
    print("2. Causal mask being too large")
    print("3. Hidden activations not being freed")
    print("4. Gradient accumulation (even with batch=1)")
    print("\nTry running with: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")