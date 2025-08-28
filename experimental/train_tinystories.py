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
    vocab_size: int = 50257   # GPT-2 vocab size
    block_size: int = 128     # Matches Reddit post
    dropout: float = 0.0
    
    # Training - Optimized for RTX 5090
    batch_size: int = 64      # Increased for RTX 5090
    gradient_accumulation_steps: int = 16  # Reduced since we increased batch_size
    max_iters: int = 10000    # ~10 epochs over TinyStories
    eval_interval: int = 500   # Matches Reddit post
    eval_iters: int = 100      # Reasonable for quick eval
    learning_rate: float = 1e-4  # Matches Reddit post
    min_lr: float = 5e-5      # Matches Reddit post
    warmup_iters: int = 1000  # Matches Reddit post
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'bfloat16'  # Can be 'float32', 'bfloat16', 'float16', 'fp8', 'fp4'
    compile: bool = False     # Torch compile (disable for now)
    
    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 5000
    checkpoint_dir: str = 'checkpoints_tinystories'
    wandb_project: str = 'tinystories-precision-test'
    wandb_run_name: Optional[str] = None
    
# -----------------------------------------------------------------------------
# Data loading

def get_batch(split: str, config: TrainingConfig, data_dir: Path = Path('data')) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data from memory-mapped dataset."""
    # Load the appropriate data
    if split == 'train':
        data = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')
    else:
        # Prefer val.bin, fall back to validation.bin for backward compatibility
        val_path = data_dir / 'val.bin'
        if not val_path.exists():
            val_path = data_dir / 'validation.bin'
        data = np.memmap(val_path, dtype=np.uint16, mode='r')
    
    # Generate random positions
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    
    # Create batch
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    
    # Move to device
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
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(iter_num: int, config: TrainingConfig) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    # Warmup
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    # Cosine decay
    if iter_num > config.warmup_iters:
        decay_ratio = (iter_num - config.warmup_iters) / (config.max_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    return config.learning_rate

# -----------------------------------------------------------------------------
# Training

def train(config: TrainingConfig):
    """Main training loop."""
    
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
        bias=False
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # Compile model if requested
    if config.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
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
        
        # Clip gradients
        if config.dtype == 'float16':
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Step optimizer
        if config.dtype == 'float16':
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = config.batch_size * config.gradient_accumulation_steps * config.block_size * local_iter_num / dt
            print(f"iter {iter_num}: loss {loss.item()*config.gradient_accumulation_steps:.4f}, "
                  f"time {dt*1000:.2f}ms, {tokens_per_sec:.0f} tok/s, lr {lr:.2e}")
            wandb_logger.log_metrics({
                'train/loss': loss.item() * config.gradient_accumulation_steps,
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
            }, step=iter_num)
            t0 = time.time()
            local_iter_num = 0
        
        # Save checkpoint periodically
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
    x = torch.tensor(tokens, dtype=torch.long, device='cuda').unsqueeze(0)
    
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
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        dtype=args.dtype,
        max_iters=args.max_iters,
        compile=args.compile,
        wandb_run_name=args.wandb_run_name
    )
    
    # Check if data exists
    data_dir = Path('data')
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
