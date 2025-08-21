#!/usr/bin/env python3
"""
Memory test for RTX 4090 (24GB) to find optimal batch size.
Run this before training to ensure settings won't cause OOM.
"""

import torch
import torch.nn as nn
import sys
import yaml
import gc
from dataclasses import fields

sys.path.append('src')

from model import create_model, ModelConfig

def format_bytes(bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def test_memory_usage():
    """Test memory usage with different batch sizes."""
    
    print("=" * 60)
    print("RTX 4090 Memory Test for Yxanul 177M")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires a GPU.")
        return
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    print(f"\nGPU: {gpu_name}")
    print(f"Total VRAM: {format_bytes(total_memory)}")
    
    # Load model config
    with open("configs/model_config.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Filter config for model creation
    valid_fields = {f.name for f in fields(ModelConfig)}
    filtered_config = {k: v for k, v in model_config["model"].items() if k in valid_fields}
    
    # Test different configurations
    test_configs = [
        {"batch_size": 1, "seq_len": 2048, "grad_checkpoint": False},
        {"batch_size": 2, "seq_len": 2048, "grad_checkpoint": False},
        {"batch_size": 4, "seq_len": 2048, "grad_checkpoint": False},
        {"batch_size": 6, "seq_len": 2048, "grad_checkpoint": False},
        {"batch_size": 8, "seq_len": 2048, "grad_checkpoint": False},
        {"batch_size": 4, "seq_len": 2048, "grad_checkpoint": True},
        {"batch_size": 8, "seq_len": 2048, "grad_checkpoint": True},
    ]
    
    print("\n" + "=" * 60)
    print("Testing different batch size configurations...")
    print("=" * 60)
    
    for config in test_configs:
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        use_grad_checkpoint = config["grad_checkpoint"]
        
        print(f"\n[Test] Batch={batch_size}, Seq={seq_len}, GradCheckpoint={use_grad_checkpoint}")
        print("-" * 40)
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Create model
            filtered_config['use_gradient_checkpointing'] = use_grad_checkpoint
            model = create_model(filtered_config)
            model = model.cuda()
            model.train()
            
            # Memory after model loading
            mem_model = torch.cuda.memory_allocated()
            print(f"  Model loaded: {format_bytes(mem_model)}")
            
            # Create optimizer (AdamW uses 2x model memory)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            mem_with_optimizer = torch.cuda.memory_allocated()
            print(f"  With optimizer: {format_bytes(mem_with_optimizer)}")
            
            # Create dummy batch
            input_ids = torch.randint(0, 50257, (batch_size, seq_len)).cuda()
            labels = input_ids.clone()
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
            
            mem_after_forward = torch.cuda.memory_allocated()
            print(f"  After forward: {format_bytes(mem_after_forward)}")
            
            # Backward pass
            loss.backward()
            
            mem_after_backward = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            print(f"  After backward: {format_bytes(mem_after_backward)}")
            print(f"  Peak memory: {format_bytes(peak_memory)}")
            
            # Calculate utilization
            utilization = (peak_memory / total_memory) * 100
            remaining = total_memory - peak_memory
            
            print(f"  VRAM utilization: {utilization:.1f}%")
            print(f"  Remaining VRAM: {format_bytes(remaining)}")
            
            # Safety check
            if utilization > 90:
                print("  [WARNING] >90% VRAM usage - risk of OOM!")
            elif utilization > 80:
                print("  [CAUTION] >80% VRAM usage - limited headroom")
            else:
                print("  [OK] Safe VRAM usage")
            
            # Cleanup
            del model, optimizer, input_ids, labels, loss, outputs
            
        except torch.cuda.OutOfMemoryError:
            print("  [ERROR] OUT OF MEMORY!")
            print("  This configuration will not work.")
        except Exception as e:
            print(f"  [ERROR] {e}")
        finally:
            # Force cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\n" + "=" * 60)
    print("Recommended Settings for RTX 4090 (24GB):")
    print("=" * 60)
    print("\nSafe Configuration (60-70% VRAM):")
    print("  - Batch size: 4")
    print("  - Sequence length: 2048")
    print("  - Gradient checkpointing: False")
    print("  - Mixed precision: BF16")
    print("  - Gradient accumulation: 8 (effective batch=32)")
    
    print("\nMemory-Optimized Configuration (if OOM):")
    print("  - Batch size: 2")
    print("  - Sequence length: 2048")
    print("  - Gradient checkpointing: True")
    print("  - Mixed precision: BF16")
    print("  - Gradient accumulation: 16 (effective batch=32)")
    
    print("\nPerformance Tips:")
    print("  1. Use BF16 instead of FP16 (better stability)")
    print("  2. Enable TF32 for matrix operations")
    print("  3. Clear cache every 100 steps")
    print("  4. Monitor with: watch -n 1 nvidia-smi")
    print("  5. If OOM during training, reduce batch_size first")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_memory_usage()