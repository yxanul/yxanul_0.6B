#!/usr/bin/env python3
"""
Training script for native TransformerEngine model with FP8.
Uses model_te.py for true FP8 acceleration.
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

# Import TE model and utilities
from model_te import TEModelConfig, TETransformerGPT, create_te_model
from wandb_logger import WandBLogger

# TransformerEngine imports
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    HAS_TE = True
except ImportError:
    print("Error: TransformerEngine required for this script!")
    print("Install with: pip install transformer-engine")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration
@dataclass
class TrainingConfig:
    # Model - GPT-2 Small with FP8-aligned dimensions
    n_layer: int = 12         
    n_head: int = 12          
    n_embd: int = 768         # Divisible by 16 ✓
    vocab_size: int = 50264   # Padded for FP8 alignment (divisible by 8)
    block_size: int = 128     # Divisible by 16 ✓
    dropout: float = 0.0
    
    # Training
    batch_size: int = 64      
    gradient_accumulation_steps: int = 16  
    max_iters: int = 1000     
    eval_interval: int = 200  
    eval_iters: int = 100      
    learning_rate: float = 1e-4  
    min_lr: float = 5e-5      
    warmup_iters: int = 1000  # Fixed warmup!
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # FP8 specific
    fp8_format: str = "E4M3"  # E4M3 for forward, E5M2 for gradients
    fp8_margin: float = 0.0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "max"
    fp8_calibration_steps: int = 20  # Calibration phase
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    compile: bool = False  # Disabled for FP8
    
    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 5000
    checkpoint_dir: str = 'checkpoints_te'
    wandb_project: str = 'tinystories-precision-test'
    wandb_run_name: Optional[str] = None

# -----------------------------------------------------------------------------
# Data loading
_TRAIN_MM = None
_VAL_MM = None

def _get_memmap(split: str, data_dir: Path) -> np.memmap:
    global _TRAIN_MM, _VAL_MM
    if split == 'train':
        if _TRAIN_MM is None:
            _TRAIN_MM = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')
        return _TRAIN_MM
    else:
        if _VAL_MM is None:
            val_path = data_dir / 'val.bin'
            if not val_path.exists():
                val_path = data_dir / 'validation.bin'
            _VAL_MM = np.memmap(val_path, dtype=np.uint16, mode='r')
        return _VAL_MM

def get_batch(split: str, config: TrainingConfig, data_dir: Path = Path('data')) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data from memory-mapped dataset."""
    data = _get_memmap('train' if split == 'train' else 'val', data_dir)
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    
    # Clamp to padded vocab size
    x = torch.clamp(x, max=config.vocab_size - 1)
    y = torch.clamp(y, max=config.vocab_size - 1)
    
    if config.device.startswith('cuda'):
        x = x.pin_memory().to(config.device, non_blocking=True)
        y = y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module, config: TrainingConfig, fp8_recipe) -> dict:
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch('validation' if split == 'val' else split, config)
            
            # Use FP8 autocast for evaluation
            with te.fp8_autocast(
                enabled=True,
                calibrating=False,  # Not calibrating during eval
                fp8_recipe=fp8_recipe
            ):
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
    """Main training loop with native TE model."""
    
    # Seeds & optimizations
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create FP8 recipe
    fp8_recipe = recipe.DelayedScaling(
        margin=config.fp8_margin,
        interval=config.fp8_interval,
        fp8_format=recipe.Format[config.fp8_format],
        amax_history_len=config.fp8_amax_history_len,
        amax_compute_algo=config.fp8_amax_compute_algo,
    )
    
    print(f"FP8 Recipe: {config.fp8_format} format, {config.fp8_interval} interval")
    
    # Initialize wandb
    wandb_logger = WandBLogger(
        project=config.wandb_project,
        run_name=config.wandb_run_name or f"tinystories-te-fp8-{config.fp8_format}",
        config=vars(config)
    )
    
    # Create native TE model
    model_config = TEModelConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_kv_heads=config.n_head,  # No GQA in TE for now
        block_size=config.block_size,
        dropout=config.dropout,
        bias=False
    )
    
    model = TETransformerGPT(model_config)
    model = model.to(config.device)
    
    print(f"\nModel configuration:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Vocab size: {config.vocab_size} (padded for FP8)")
    print(f"  Hidden size: {config.n_embd}")
    print(f"  FFN size: {model_config.ffn_hidden_size}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # Training state
    best_val_loss = float('inf')
    iter_num = 0
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    print(f"\nStarting FP8 training with TransformerEngine")
    print(f"Iterations: {config.max_iters}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Tokens per iteration: {config.batch_size * config.gradient_accumulation_steps * config.block_size:,}")
    print(f"Calibration steps: {config.fp8_calibration_steps}")
    print("-" * 50)
    
    t0 = time.time()
    local_iter_num = 0
    
    while iter_num < config.max_iters:
        # Determine if we're calibrating
        calibrating = iter_num < config.fp8_calibration_steps
        
        # Evaluate periodically  
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config, fp8_recipe)
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb_logger.log_eval(losses, step=iter_num)
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                # Save without FP8 recipe (it's not pickleable)
                torch.save(checkpoint, Path(config.checkpoint_dir) / 'best_model.pt')
                print(f"Saved new best model (val loss: {best_val_loss:.4f})")
        
        # Adjust learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training step with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = get_batch('train', config)
            
            # Forward pass with FP8
            with te.fp8_autocast(
                enabled=True,
                calibrating=calibrating,
                fp8_recipe=fp8_recipe
            ):
                logits, loss = model(X, Y)
            
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iter_num % config.log_interval == 0:
            local_iter_num = max(1, local_iter_num)
            t1 = time.time()
            dt = t1 - t0
            tokens_per_iter = config.batch_size * config.gradient_accumulation_steps * config.block_size
            tokens_per_sec = tokens_per_iter * local_iter_num / dt
            
            status = "[FP8-TE]"
            if calibrating:
                status += f" (calibrating {iter_num}/{config.fp8_calibration_steps})"
            
            print(f"iter {iter_num}: loss {loss.item()*config.gradient_accumulation_steps:.4f}, "
                  f"{tokens_per_sec:.0f} tok/s, lr {lr:.2e} {status}")
            
            wandb_logger.log_metrics({
                'train/loss': loss.item() * config.gradient_accumulation_steps,
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
                'train/calibrating': calibrating,
            }, step=iter_num)
            
            t0 = time.time()
            local_iter_num = 0
        
        iter_num += 1
        local_iter_num += 1
    
    # Final evaluation
    print("\nTraining complete!")
    losses = estimate_loss(model, config, fp8_recipe)
    print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Perplexity: {math.exp(best_val_loss):.2f}")
    
    wandb_logger.set_summary(
        best_val_loss=best_val_loss,
        final_train_loss=losses['train'],
        final_val_loss=losses['val'],
        perplexity=math.exp(best_val_loss)
    )
    wandb_logger.finish()
    
    return model, best_val_loss

# -----------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iters', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--fp8_format', type=str, default='E4M3',
                       choices=['E4M3', 'E5M2', 'HYBRID'],
                       help='FP8 format (E4M3 for compute, E5M2 for gradients)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Custom wandb run name')
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        max_iters=args.max_iters,
        fp8_format=args.fp8_format,
        wandb_run_name=args.wandb_run_name
    )
    
    # Check data exists
    data_dir = Path('data')
    if not (data_dir / 'train.bin').exists():
        print("Data not found! Please run prepare_tinystories_local.py first")
        sys.exit(1)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA required for TransformerEngine FP8 training!")
        sys.exit(1)
    
    # Train
    model, best_loss = train(config)
    
    print("\n" + "="*50)
    print("FP8 Training with TransformerEngine Complete!")
    print("="*50)