#!/usr/bin/env python3
"""
TinyStories training script with precision comparison capabilities.
Based on the Reddit Gemma3 270M implementation but optimized for RTX 5090.
"""

import os
import sys
import time
import math
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# Import our model and logger
from model import ModelConfig, SimpleGPT as GPT
from wandb_logger import WandBLogger

# -----------------------------------------------------------------------------
# Configuration
@dataclass
class TrainingConfig:
    # Model - GPT-2 Small (124M parameters)
    n_layer: int = 12         # GPT-2 small
    n_head: int = 12          # GPT-2 small
    n_embd: int = 768         # GPT-2 small
    vocab_size: int = 50257   # GPT-2 vocab (50257), SmolLM (49152), or SuperBPE (200005)
    block_size: int = 128     # Matches Reddit post
    dropout: float = 0.05  # Conservative dropout for regularization
    use_factorized_embedding: bool = False  # Enable factorized embeddings
    embedding_rank: int = 128  # Rank for factorization
    
    # Training - Optimized for RTX 5090
    batch_size: int = 64      # Increased for RTX 5090
    gradient_accumulation_steps: int = 16  # Reduced since we increased batch_size
    max_iters: int = 10000    # ~10 epochs over TinyStories
    eval_interval: int = 200  # More frequent eval for better tracking   # Matches Reddit post
    eval_iters: int = 100      # Reasonable for quick eval
    learning_rate: float = 3e-3  # Proven stable, 3x faster convergence
    min_lr: float = 3e-4      # Keep same ratio (10x reduction)
    warmup_iters: int = 1000  # Matches Reddit post
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'bfloat16'  # Can be 'float32', 'bfloat16', 'float16', 'fp8', 'fp4'
    compile: bool = False     # Torch compile (disable for now)
    
    # Logging
    log_interval: int = 200  # Reduced frequency to minimize overhead
    checkpoint_interval: int = 500  # Less frequent checkpoints for performance
    checkpoint_dir: str = 'checkpoints_tinystories'
    wandb_project: str = 'tinystories-precision-test'
    wandb_run_name: Optional[str] = None
    
    # Data configuration
    data_dir: str = 'data'  # Change to 'data_superbpe' for SuperBPE
    use_superbpe: bool = False  # Set True to use SuperBPE tokenizer
    
# -----------------------------------------------------------------------------
# Data loading (reuse memmaps for speed)
_TRAIN_MM = None
_VAL_MM = None

def _get_memmap(split: str, data_dir: Path, vocab_size: int = 50257) -> np.memmap:
    global _TRAIN_MM, _VAL_MM
    # CRITICAL FIX: Use uint32 for vocabularies > 65535 (uint16 max)
    dtype = np.uint32 if vocab_size > 65535 else np.uint16
    
    if split == 'train':
        if _TRAIN_MM is None:
            _TRAIN_MM = np.memmap(data_dir / 'train.bin', dtype=dtype, mode='r')
        return _TRAIN_MM
    else:
        if _VAL_MM is None:
            val_path = data_dir / 'val.bin'
            if not val_path.exists():
                val_path = data_dir / 'validation.bin'
            _VAL_MM = np.memmap(val_path, dtype=dtype, mode='r')
        return _VAL_MM


def get_batch(split: str, config: TrainingConfig, data_dir: Path = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data from memory-mapped dataset - OPTIMIZED VERSION."""
    if data_dir is None:
        data_dir = Path(config.data_dir)
    data = _get_memmap('train' if split == 'train' else 'val', data_dir, config.vocab_size)
    
    # Generate random positions
    ix = torch.randint(len(data) - config.block_size - 1, (config.batch_size,))
    
    # VECTORIZED: Create all indices at once
    offsets = torch.arange(config.block_size + 1)
    indices = ix.unsqueeze(1) + offsets.unsqueeze(0)  # [batch_size, block_size + 1]
    
    # Fetch all data in one operation
    batch_data = torch.from_numpy(data[indices.numpy()].astype(np.int64))
    
    # Split into x and y
    x = batch_data[:, :-1]  # [batch_size, block_size]
    y = batch_data[:, 1:]   # [batch_size, block_size]
    
    # Move to device (no pin_memory after creation - that's wrong)
    if config.device.startswith('cuda'):
        x = x.to(config.device, non_blocking=True)
        y = y.to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module, config: TrainingConfig) -> dict:
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch('validation' if split == 'val' else split, config)
            device_type = 'cuda' if config.device.startswith('cuda') else 'cpu'
            amp_dtype = torch.bfloat16 if config.dtype == 'bfloat16' else (torch.float16 if config.dtype == 'float16' else torch.float32)
            ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype) if device_type == 'cuda' else nullcontext()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def compute_layer_gradient_stats(model):
    """Compute per-layer gradient statistics for monitoring."""
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_mean = param.grad.data.mean().item()
            grad_std = param.grad.data.std().item()
            
            # Get layer type from name
            if 'embed' in name:
                layer_type = 'embedding'
            elif 'ln' in name or 'norm' in name:
                layer_type = 'norm'
            elif 'attn' in name:
                layer_type = 'attention'
            elif 'ffn' in name or 'mlp' in name:
                layer_type = 'ffn'
            elif 'lm_head' in name:
                layer_type = 'lm_head'
            else:
                layer_type = 'other'
            
            if layer_type not in layer_stats:
                layer_stats[layer_type] = {
                    'grad_norm': [],
                    'grad_mean': [],
                    'grad_std': []
                }
            
            layer_stats[layer_type]['grad_norm'].append(grad_norm)
            layer_stats[layer_type]['grad_mean'].append(grad_mean)
            layer_stats[layer_type]['grad_std'].append(grad_std)
    
    # Average stats per layer type
    avg_stats = {}
    for layer_type, stats in layer_stats.items():
        avg_stats[layer_type] = {
            'grad_norm': sum(stats['grad_norm']) / len(stats['grad_norm']),
            'grad_mean': sum(stats['grad_mean']) / len(stats['grad_mean']),
            'grad_std': sum(stats['grad_std']) / len(stats['grad_std'])
        }
    
    return avg_stats

def get_lr(iter_num: int, config: TrainingConfig) -> float:
    """Learning rate schedule with warmup, plateau, and cosine decay.
    
    Schedule:
    - 0 to warmup_iters: Linear warmup
    - warmup_iters to 80% of max_iters: Constant at peak LR
    - 80% to 100% of max_iters: Cosine decay to min_lr
    """
    # Warmup phase
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    
    # Plateau phase - maintain peak LR for most of training
    plateau_end = int(0.8 * config.max_iters)
    if iter_num < plateau_end:
        return config.learning_rate
    
    # Cosine decay only in final 20%
    decay_iters = config.max_iters - plateau_end
    decay_ratio = (iter_num - plateau_end) / decay_iters
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# -----------------------------------------------------------------------------
# Training

def train(config: TrainingConfig):
    """Main training loop."""
    # Seeds & TF32 for speed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Initialize wandb
    wandb_logger = WandBLogger(
        project=config.wandb_project,
        run_name=config.wandb_run_name or f"tinystories-{config.dtype}",
        config=vars(config)
    )
    
    # Create model
    model_config = ModelConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_kv_heads=config.n_head // 4,  # GQA with 4x compression
        block_size=config.block_size,
        dropout=config.dropout,
        bias=False,
        use_factorized_embedding=config.use_factorized_embedding,
        embedding_rank=config.embedding_rank
    )
    
    model = GPT(model_config)
    model = model.to(config.device)
    print(f"Created model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Setup precision mode
    if config.dtype == 'fp8':
        print("FP8 mode: Requires TransformerEngine or custom implementation")
        # TODO: Add FP8 support
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    elif config.dtype == 'fp4':
        print("FP4/NVF4 mode: Requires RTX 5090 specific implementation")
        # TODO: Add FP4 support for RTX 5090
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    elif config.dtype == 'bfloat16':
        print("BF16 mode: Using native PyTorch autocast")
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    elif config.dtype == 'float16':
        print("FP16 mode: Using native PyTorch autocast with GradScaler")
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler()
    else:
        print("FP32 mode: No mixed precision")
        ctx = nullcontext()
    
    # Optimizer - using fused version for ~5% speedup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=True  # Fused optimizer reduces kernel launches
    )
    
    # Compile model if requested
    if config.compile:
        print("Compiling model with max-autotune (takes ~60s but worth it)...")
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    
    # Training loop
    best_val_loss = float('inf')
    iter_num = 0
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    print(f"\nStarting training for {config.max_iters} iterations")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Tokens per iteration: {config.batch_size * config.gradient_accumulation_steps * config.block_size:,}")
    print(f"Total tokens to train on: {config.max_iters * config.batch_size * config.gradient_accumulation_steps * config.block_size:,}")
    print("-" * 50)
    
    t0 = time.time()
    local_iter_num = 0
    
    while iter_num < config.max_iters:
        # Evaluate periodically
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config)
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb_logger.log_eval(losses, step=iter_num)
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                torch.save(checkpoint, Path(config.checkpoint_dir) / 'best_model.pt')
                print(f"Saved new best model (val loss: {best_val_loss:.4f})")
        
        # Adjust learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Accumulate gradients
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = get_batch('train', config)
            
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / config.gradient_accumulation_steps
            
            if config.dtype == 'float16':
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Compute gradient statistics before clipping
        grad_norm_before_clip = 0.0
        param_norm = 0.0
        update_norm = 0.0
        num_params_with_grad = 0
        
        # Unscale gradients if using AMP
        if config.dtype == 'float16':
            scaler.unscale_(optimizer)
        
        # Calculate gradient norm and parameter norm
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_before_clip += p.grad.data.norm(2).item() ** 2
                param_norm += p.data.norm(2).item() ** 2
                num_params_with_grad += 1
        
        grad_norm_before_clip = grad_norm_before_clip ** 0.5
        param_norm = param_norm ** 0.5
        
        # Clip gradients and get the norm after clipping
        grad_norm_after_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        
        # Check if clipping occurred
        grad_clipped = grad_norm_after_clip < grad_norm_before_clip - 1e-6
        
        # Store old parameters to compute update norm
        old_params = {name: p.data.clone() for name, p in model.named_parameters() if p.grad is not None}
        
        # Step optimizer
        if config.dtype == 'float16':
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Compute parameter update norm
        for name, p in model.named_parameters():
            if p.grad is not None and name in old_params:
                update_norm += (p.data - old_params[name]).norm(2).item() ** 2
        update_norm = update_norm ** 0.5
        
        # Compute update ratio
        update_ratio = update_norm / (param_norm + 1e-8)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if iter_num % config.log_interval == 0:
            # Count this iteration in the window
            local_iter_num = max(1, local_iter_num)
            t1 = time.time()
            dt = t1 - t0
            tokens_per_iter = config.batch_size * config.gradient_accumulation_steps * config.block_size
            tokens_per_sec = tokens_per_iter * local_iter_num / dt
            avg_ms_per_iter = (dt / local_iter_num) * 1000.0
            print(f"iter {iter_num}: loss {loss.item()*config.gradient_accumulation_steps:.4f}, "
                  f"window {dt*1000:.2f}ms ({avg_ms_per_iter:.2f} ms/iter), {tokens_per_sec:.0f} tok/s, lr {lr:.2e}")
            
            # Log gradient statistics less frequently to reduce overhead
            if iter_num % 500 == 0 and iter_num > 0:  # Much less frequent
                print(f"  grad_norm: {grad_norm_after_clip:.4f} (pre-clip: {grad_norm_before_clip:.4f}, clipped: {grad_clipped})")
                print(f"  update_ratio: {update_ratio:.6f} (update_norm: {update_norm:.4f}, param_norm: {param_norm:.2f})")
            
            # Compute and log per-layer stats rarely (expensive operation)
            layer_grad_metrics = {}
            if iter_num % 1000 == 0 and iter_num > 0:  # Very rare - only for debugging
                layer_stats = compute_layer_gradient_stats(model)
                for layer_type, stats in layer_stats.items():
                    print(f"  {layer_type}: grad_norm={stats['grad_norm']:.4f}, std={stats['grad_std']:.6f}")
                    layer_grad_metrics[f'gradients/{layer_type}_norm'] = stats['grad_norm']
                    layer_grad_metrics[f'gradients/{layer_type}_std'] = stats['grad_std']
            
            wandb_logger.log_metrics({
                'train/loss': loss.item() * config.gradient_accumulation_steps,
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
                'train/avg_ms_per_iter': avg_ms_per_iter,
                'gradients/norm_before_clip': grad_norm_before_clip,
                'gradients/norm_after_clip': grad_norm_after_clip,
                'gradients/clipped': float(grad_clipped),
                'gradients/update_ratio': update_ratio,
                'gradients/update_norm': update_norm,
                'gradients/param_norm': param_norm,
                **layer_grad_metrics  # Merge per-layer stats when available
            }, step=iter_num)
            t0 = time.time()
            local_iter_num = 0
        
        # Save checkpoint periodically (less frequent for performance)
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config
            }
            torch.save(checkpoint, Path(config.checkpoint_dir) / f'ckpt_{iter_num}.pt')
            print(f"Saved checkpoint at iteration {iter_num}")
        
        iter_num += 1
        local_iter_num += 1
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final evaluation
    losses = estimate_loss(model, config)
    print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    wandb_logger.log_eval(losses, step=iter_num)
    wandb_logger.set_summary(
        best_val_loss=best_val_loss,
        final_train_loss=losses['train'],
        final_val_loss=losses['val']
    )
    wandb_logger.finish()
    
    return model, best_val_loss

# -----------------------------------------------------------------------------
# Generation

def generate(model: nn.Module, prompt: str, max_tokens: int = 200, temperature: float = 0.8, top_k: int = 50):
    """Generate text from a prompt."""
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    tokens = enc.encode(prompt)
    device = next(model.parameters()).device
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits for last position
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append and continue
            x = torch.cat((x, idx_next), dim=1)
            
            # Stop at EOS
            if idx_next.item() == enc.eot_token:
                break
    
    # Decode
    generated = x[0].tolist()
    return enc.decode(generated)

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='bfloat16', 
                       choices=['float32', 'float16', 'bfloat16', 'fp8', 'fp4'],
                       help='Precision mode for training')
    parser.add_argument('--max_iters', type=int, default=10000,
                       help='Number of training iterations')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile()')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Custom wandb run name')
    parser.add_argument('--factorized', action='store_true',
                       help='Use factorized embeddings (reduces model by ~32M params)')
    parser.add_argument('--embedding_rank', type=int, default=128,
                       help='Rank for factorized embeddings (default: 128)')
    parser.add_argument('--superbpe', action='store_true',
                       help='Use SuperBPE tokenizer (40% fewer tokens)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Custom data directory (e.g., data_textbooks_superbpe)')
    parser.add_argument('--vocab_size', type=int, default=None,
                       help='Custom vocabulary size (e.g., 200005 for SuperBPE)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (default: 3e-3, proven stable up to 3e-3)')
    parser.add_argument('--block_size', type=int, default=None,
                       help='Context length (default: 128, recommended: 2048 for A100)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: auto-selected based on vocab, recommended: 64 for A100)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                       help='Gradient accumulation steps (default: 16-32 based on vocab)')
    args = parser.parse_args()
    
    # Create config
    # Determine data directory and vocab size
    if args.data_dir:
        # Custom data directory specified
        data_dir = args.data_dir
        vocab_size = args.vocab_size if args.vocab_size else (200005 if 'superbpe' in args.data_dir else 50257)
    elif args.superbpe:
        # SuperBPE flag used
        data_dir = 'data_superbpe'
        vocab_size = 200005
    else:
        # Default GPT-2
        data_dir = 'data'
        vocab_size = 50257
    
    # Override vocab size if explicitly provided
    if args.vocab_size:
        vocab_size = args.vocab_size
    
    # Adjust settings for large vocabulary (200k) to avoid OOM
    if vocab_size > 100000:
        # Settings for large vocabulary (SuperBPE)
        batch_size = args.batch_size if args.batch_size else 32  # Middle ground for 24GB VRAM
        block_size = args.block_size if args.block_size else 128  # Keep original for good throughput
        gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps else 32  # Maintain 1024 effective batch
    else:
        # Settings for normal vocabulary (GPT-2 or SmolLM)
        batch_size = args.batch_size if args.batch_size else 64
        block_size = args.block_size if args.block_size else 128
        gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps else 16
    
    config = TrainingConfig(
        dtype=args.dtype,
        max_iters=args.max_iters,
        compile=args.compile,
        wandb_run_name=args.wandb_run_name,
        use_factorized_embedding=args.factorized,
        embedding_rank=args.embedding_rank,
        use_superbpe=args.superbpe or vocab_size > 100000,
        data_dir=data_dir,
        vocab_size=vocab_size,
        batch_size=batch_size,
        block_size=block_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate if args.learning_rate else 3e-3  # Default to proven stable 3e-3
    )
    
    # Check if data exists
    data_dir = Path(config.data_dir)
    if not (data_dir / 'train.bin').exists():
        print("Data not found! Please run prepare_tinystories.py first")
        sys.exit(1)
    
    # Train
    model, best_loss = train(config)
    
    # Test generation
    print("\n" + "="*50)
    print("Testing generation...")
    prompt = "Once upon a time there was a little"
    generated = generate(model, prompt, max_tokens=100)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
