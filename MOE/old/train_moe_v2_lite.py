"""
Lite version of V2 training script with memory optimizations.
For debugging OOM issues.
"""

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import argparse
import gc

from model_moe_v2 import MoEConfig, MoEModelV2

def main():
    parser = argparse.ArgumentParser(description="Lightweight V2 MoE training")
    
    # Reduced model config for testing
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_embd", type=int, default=896, help="Embedding dimension")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")
    parser.add_argument("--expert_expansion", type=float, default=3.0, help="Expert expansion")
    parser.add_argument("--capacity_factor", type=float, default=1.0, help="Capacity factor")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps")
    parser.add_argument("--use_amp", action="store_true", help="Use BF16 mixed precision")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data_mixed_3b", help="Data directory")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Lightweight V2 MoE Training (Memory Debug)")
    print("="*60)
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_mem:.1f} GB")
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create config
    config = MoEConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=28,
        n_kv_head=7,
        num_experts=args.num_experts,
        expert_expansion=args.expert_expansion,
        capacity_factor=args.capacity_factor,
        block_size=args.seq_len,  # Match sequence length
        dropout=0.0,  # No dropout for testing
    )
    
    print(f"\nModel Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Expert expansion: {config.expert_expansion}x")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    
    # Create model
    print("\nCreating model...")
    model = MoEModelV2(config)
    
    if args.use_amp:
        print("Converting to BF16...")
        model = model.bfloat16()
    
    model = model.cuda()
    
    # Memory after model
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\nMemory after model creation:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    
    # Create optimizer with minimal memory
    print("\nCreating optimizer...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, fused=True)  # Fused optimizer saves memory
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    try:
        train_data = np.memmap(
            os.path.join(args.data_dir, "train.bin"),
            dtype=np.uint16,
            mode='r'
        )
        print(f"  Loaded {len(train_data) / 1e6:.1f}M tokens")
    except Exception as e:
        print(f"  Warning: Could not load data: {e}")
        print("  Using random data for testing")
        train_data = None
    
    # Training loop
    print("\nStarting training...")
    print("-"*40)
    
    model.train()
    
    for step in range(args.steps):
        # Get batch
        if train_data is not None:
            # Sample from real data
            ix = torch.randint(len(train_data) - args.seq_len, (args.batch_size,))
            x = torch.stack([
                torch.from_numpy(train_data[i:i+args.seq_len].astype(np.int64))
                for i in ix
            ]).cuda()
        else:
            # Random data
            x = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len)).cuda()
        
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        try:
            if args.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, x)
            else:
                logits, loss = model(x, x)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Memory stats
            if step % 5 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                
                print(f"Step {step:3d} | Loss: {loss.item():.4f} | "
                      f"Mem: {allocated:.2f}/{reserved:.2f} GB (peak: {max_allocated:.2f} GB)")
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n✗ OOM at step {step}!")
            print(f"  Error: {e}")
            
            # Get detailed memory info
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
            print(f"\nMemory at OOM:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Peak: {max_allocated:.2f} GB")
            
            # Try to get memory summary
            try:
                print("\nMemory Summary:")
                print(torch.cuda.memory_summary())
            except:
                pass
            
            break
        
        except Exception as e:
            print(f"\n✗ Error at step {step}: {e}")
            break
    
    else:
        print("\n✓ Training completed successfully!")
        
        # Final stats
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"\nFinal Memory Stats:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Peak: {max_allocated:.2f} GB")
        
        # Estimate tokens/sec
        if args.steps > 0:
            print(f"\nTokens per step: {args.batch_size * args.seq_len}")
            print("(Run with more steps for throughput measurement)")


if __name__ == "__main__":
    # Set memory config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    main()