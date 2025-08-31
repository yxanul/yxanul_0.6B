#!/usr/bin/env python3
"""
CLEAN FP8 training script - the ACTUAL fastest version.
This uses the simplified model without dead optimizations.

Key insights:
- NO FP8 weight caching (overhead > benefit at 112M scale)
- NO gradient fusion (adds memory traffic)
- Simple is faster for small models

Achieves 196k tokens/sec on RTX 5090.
"""

import os
import math
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import argparse

# Import CLEAN model (the actually fast one)
from model_te_clean import ModelConfig, CleanGPT_TE, get_fp8_recipe

# TransformerEngine
import transformer_engine.pytorch as te

# Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Note: wandb not available")


@dataclass
class TrainingConfig:
    # Model config
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 49152
    n_kv_heads: int = 3
    block_size: int = 2048
    dropout: float = 0.05
    
    # FP8 config
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_warmup_steps: int = 100
    
    # Optimizations - simplified for CLEAN version
    fuse_wgrad_accumulation: bool = False  # DISABLED - adds overhead
    cache_fp8_weights: bool = False  # NOT IMPLEMENTED - overhead > benefit
    
    # Training config
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    lr_decay_iters: int = 2000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # System
    device: str = 'cuda'
    compile: bool = False  # Disabled for TE compatibility
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = 'checkpoints_fp8_optimized'
    data_dir: str = 'data_mixed_3b'
    
    # Wandb
    wandb_project: str = 'fp8-optimized'
    wandb_run_name: Optional[str] = None


class DataLoader:
    """Memory-mapped data loader."""
    def __init__(self, data_dir, block_size, device='cuda'):
        self.block_size = block_size
        self.device = device
        
        data_path = Path(data_dir)
        self.train_data = np.memmap(data_path / 'train.bin', dtype=np.uint16, mode='r')
        self.val_data = np.memmap(data_path / 'val.bin', dtype=np.uint16, mode='r')
        
        print(f"Loaded data from {data_dir}")
        print(f"  Train: {len(self.train_data):,} tokens")
        print(f"  Val: {len(self.val_data):,} tokens")
    
    def get_batch(self, split, batch_size):
        data = self.train_data if split == 'train' else self.val_data
        max_start = len(data) - self.block_size - 1
        ix = torch.randint(max_start, (batch_size,))
        
        offsets = torch.arange(self.block_size + 1)
        indices = ix.unsqueeze(1) + offsets.unsqueeze(0)
        batch_data = torch.from_numpy(data[indices.numpy()].astype(np.int64)).pin_memory()
        
        x = batch_data[:, :-1].to(self.device, non_blocking=True)
        y = batch_data[:, 1:].to(self.device, non_blocking=True)
        
        return x, y


def get_lr(iter_num, config):
    """Learning rate schedule."""
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    
    plateau_iters = int(config.lr_decay_iters * 0.6)
    if iter_num < config.warmup_iters + plateau_iters:
        return config.learning_rate
    
    decay_iter = iter_num - config.warmup_iters - plateau_iters
    decay_length = config.lr_decay_iters - config.warmup_iters - plateau_iters
    
    if decay_iter > decay_length:
        return config.min_lr
    
    decay_ratio = decay_iter / decay_length
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def evaluate(model, data_loader, config, fp8_recipe):
    """Evaluate model."""
    model.eval()
    losses = []
    
    for _ in range(config.eval_iters):
        x, y = data_loader.get_batch('val', config.batch_size)
        
        if config.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def save_checkpoint(model, optimizer, config, iter_num, val_loss, checkpoint_path):
    """Save checkpoint."""
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': asdict(config),
        'iter_num': iter_num,
        'val_loss': val_loss,
    }
    
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)


def train():
    """Main training loop with optimizations."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='data_mixed_3b', help='Data directory')
    parser.add_argument('--max_iters', type=int, default=2000, help='Max iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='Eval interval')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--no_fp8', action='store_true', help='Disable FP8')
    parser.add_argument('--no_fusion', action='store_true', help='Disable gradient fusion (always disabled in CLEAN)')
    # Removed --no_caching as it's not implemented in CLEAN version
    args = parser.parse_args()
    
    # Configuration
    config = TrainingConfig()
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_fp8:
        config.use_fp8 = False
    # Gradient fusion always disabled in CLEAN version
    config.fuse_wgrad_accumulation = False
    config.cache_fp8_weights = False
    
    config.data_dir = args.data_dir
    config.max_iters = args.max_iters
    config.eval_interval = args.eval_interval
    config.log_interval = args.log_interval
    
    # Model configuration
    model_config = ModelConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_kv_heads=config.n_kv_heads,
        block_size=config.block_size,
        dropout=config.dropout,
        use_fp8=config.use_fp8,
        fp8_amax_history_len=config.fp8_amax_history_len,
        fuse_wgrad_accumulation=False,  # Always disabled in CLEAN
    )
    
    # Create CLEAN model (the fast one)
    model = CleanGPT_TE(model_config).to(config.device)
    model = model.to(torch.bfloat16)
    
    # Get FP8 recipe
    fp8_recipe = get_fp8_recipe(model_config)
    
    # Optimizer - always use standard setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=True
    )
    
    # Data loader
    data_loader = DataLoader(config.data_dir, config.block_size, config.device)
    
    # Wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"fp8_optimized_{config.n_layer}L",
            config=asdict(config)
        )
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    # Training info
    print("\n" + "="*50)
    print("Starting CLEAN FP8 Training (Actual Fastest)")
    print("="*50)
    print(f"Model: {model_config.n_layer}L, {model_config.n_head}H, {model_config.n_embd}D")
    print(f"Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"Status:")
    print(f"  - FP8: {config.use_fp8}")
    print(f"  - Gradient fusion: DISABLED (overhead)")
    print(f"  - Weight caching: NOT IMPLEMENTED (overhead > benefit)")
    print(f"  - Expected: 196k tokens/sec on RTX 5090")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print("="*50 + "\n")
    
    model.train()
    iter_num = 0
    best_val_loss = float('inf')
    t0 = time.time()
    tokens_processed = 0
    
    for iter_num in range(config.max_iters):
        # Learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Zero gradients - simple and clean
        optimizer.zero_grad(set_to_none=True)  # Faster than zeroing
        
        # Accumulate gradients
        total_loss = 0
        
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = data_loader.get_batch('train', config.batch_size)
            
            # Determine if using FP8
            use_fp8_now = config.use_fp8 and (iter_num >= config.fp8_warmup_steps)
            
            # Forward pass - CLEAN version without is_first_microbatch
            if use_fp8_now:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, loss = model(x, y)  # No is_first_microbatch!
            else:
                logits, loss = model(x, y)
            
            total_loss += loss.item()
            loss = loss / config.gradient_accumulation_steps
            
            # Backward pass - gradients accumulate naturally
            loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        
        # Optimizer step - simple and clean
        optimizer.step()
        
        # Update token count
        tokens_processed += config.batch_size * config.block_size * config.gradient_accumulation_steps
        
        # Logging
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = tokens_processed / dt if dt > 0 else 0
            
            avg_loss = total_loss / config.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {avg_loss:.4f}, lr {lr:.2e}, "
                  f"{tokens_per_sec/1e3:.1f}k tok/s, FP8: {use_fp8_now}")
            
            if WANDB_AVAILABLE:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/lr': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/grad_norm': grad_norm.item() if config.grad_clip > 0 else 0,
                    'train/fp8_active': use_fp8_now,
                }, step=iter_num)
            
            tokens_processed = 0
            t0 = time.time()
        
        # Evaluation
        if iter_num % config.eval_interval == 0:
            val_loss = evaluate(model, data_loader, config, fp8_recipe)
            print(f"Step {iter_num}: val loss {val_loss:.4f}")
            
            if WANDB_AVAILABLE:
                wandb.log({'val/loss': val_loss}, step=iter_num)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, config, iter_num, val_loss,
                    Path(config.checkpoint_dir) / 'best_model_fp8_optimized.pt'
                )
        
        # Regular checkpoints
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            save_checkpoint(
                model, optimizer, config, iter_num, val_loss,
                Path(config.checkpoint_dir) / f'checkpoint_{iter_num}_fp8_optimized.pt'
            )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("CLEAN implementation - simplicity wins at this scale!")
    print("="*50)


if __name__ == "__main__":
    train()