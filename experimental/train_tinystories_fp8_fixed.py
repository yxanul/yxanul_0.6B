#!/usr/bin/env python3
"""
TinyStories training with FP8 - Fixed for tensor alignment.
FP8 requires dimensions divisible by 8 (height) and 16 (width).
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
    # Model - GPT-2 Small (124M parameters) with FP8-aligned dimensions
    n_layer: int = 12         
    n_head: int = 12          
    n_embd: int = 768         # Divisible by 16 ✓
    vocab_size: int = 50264   # Padded from 50257 to nearest multiple of 8
    block_size: int = 128     # Divisible by 16 ✓
    dropout: float = 0.0
    
    # Training - Optimized for RTX 5090
    batch_size: int = 64      # Divisible by 8 ✓
    gradient_accumulation_steps: int = 16  
    max_iters: int = 1000     
    eval_interval: int = 200  # More frequent eval
    eval_iters: int = 100      
    learning_rate: float = 1e-4  
    min_lr: float = 5e-5      
    warmup_iters: int = 100    
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'fp8'  # Will test FP8 directly
    compile: bool = False     
    
    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 5000
    checkpoint_dir: str = 'checkpoints_tinystories_fp8'
    wandb_project: str = 'tinystories-precision-test'
    wandb_run_name: Optional[str] = None

# -----------------------------------------------------------------------------
# Data loading (reuse from original)
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
    
    # Clamp values to padded vocab size (in case tokenizer produced 50257-50263)
    x = torch.clamp(x, max=config.vocab_size - 1)
    y = torch.clamp(y, max=config.vocab_size - 1)
    
    if config.device.startswith('cuda'):
        x = x.pin_memory().to(config.device, non_blocking=True)
        y = y.pin_memory().to(config.device, non_blocking=True)
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
            
            # Try FP8 if available, otherwise BF16
            try:
                import transformer_engine.pytorch as te
                with te.fp8_autocast(enabled=True):
                    logits, loss = model(X, Y)
            except:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(X, Y)
            
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(iter_num: int, config: TrainingConfig) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    if iter_num > config.warmup_iters:
        decay_ratio = (iter_num - config.warmup_iters) / (config.max_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    return config.learning_rate

# -----------------------------------------------------------------------------
# Training
def train(config: TrainingConfig):
    """Main training loop testing FP8."""
    
    # Seeds & optimizations
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Check for TransformerEngine
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        HAS_TE = True
        print("TransformerEngine detected - FP8 training available")
        
        # Create FP8 recipe
        fp8_recipe = recipe.DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=recipe.Format.E4M3,
            amax_history_len=1024,
            amax_compute_algo="max",
        )
    except ImportError:
        HAS_TE = False
        print("TransformerEngine not found - falling back to BF16")
        fp8_recipe = None
    
    # Initialize wandb
    run_name = config.wandb_run_name or f"tinystories-{'fp8' if HAS_TE else 'bf16'}"
    wandb_logger = WandBLogger(
        project=config.wandb_project,
        run_name=run_name,
        config=vars(config)
    )
    
    # Create model with FP8-aligned dimensions
    model_config = ModelConfig(
        vocab_size=config.vocab_size,  # 50264 (padded)
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_kv_heads=config.n_head // 4,  
        block_size=config.block_size,
        dropout=config.dropout,
        bias=False
    )
    
    model = GPT(model_config)
    model = model.to(config.device)
    
    # Print actual vocab size being used
    print(f"Created model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"Vocab size: {config.vocab_size} (padded from 50257 for FP8 alignment)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    iter_num = 0
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    precision_str = "FP8" if HAS_TE else "BF16"
    print(f"\nStarting {precision_str} training for {config.max_iters} iterations")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Tokens per iteration: {config.batch_size * config.gradient_accumulation_steps * config.block_size:,}")
    print("-" * 50)
    
    t0 = time.time()
    local_iter_num = 0
    
    # FP8 calibration phase
    calibrating = True
    calibration_steps = 10
    
    while iter_num < config.max_iters:
        # Evaluate periodically
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config)
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
                torch.save(checkpoint, Path(config.checkpoint_dir) / 'best_model.pt')
                print(f"Saved new best model (val loss: {best_val_loss:.4f})")
        
        # Adjust learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Stop calibrating after initial steps
        if HAS_TE and iter_num >= calibration_steps:
            calibrating = False
        
        # Accumulate gradients
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = get_batch('train', config)
            
            # Forward pass with appropriate precision
            if HAS_TE:
                with te.fp8_autocast(
                    enabled=True,
                    calibrating=calibrating,
                    fp8_recipe=fp8_recipe
                ):
                    logits, loss = model(X, Y)
            else:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(X, Y)
            
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Step optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if iter_num % config.log_interval == 0:
            local_iter_num = max(1, local_iter_num)
            t1 = time.time()
            dt = t1 - t0
            tokens_per_iter = config.batch_size * config.gradient_accumulation_steps * config.block_size
            tokens_per_sec = tokens_per_iter * local_iter_num / dt
            
            status = f"[{precision_str}]"
            if HAS_TE and calibrating:
                status += " (calibrating)"
            
            print(f"iter {iter_num}: loss {loss.item()*config.gradient_accumulation_steps:.4f}, "
                  f"{tokens_per_sec:.0f} tok/s, lr {lr:.2e} {status}")
            
            wandb_logger.log_metrics({
                'train/loss': loss.item() * config.gradient_accumulation_steps,
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
                'train/precision': precision_str,
            }, step=iter_num)
            
            t0 = time.time()
            local_iter_num = 0
        
        iter_num += 1
        local_iter_num += 1
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Perplexity: {math.exp(best_val_loss):.2f}")
    
    # Final evaluation
    losses = estimate_loss(model, config)
    print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    wandb_logger.set_summary(
        best_val_loss=best_val_loss,
        final_train_loss=losses['train'],
        final_val_loss=losses['val'],
        precision=precision_str
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
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Custom wandb run name')
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        max_iters=args.max_iters,
        wandb_run_name=args.wandb_run_name
    )
    
    # Check if data exists
    data_dir = Path('data')
    if not (data_dir / 'train.bin').exists():
        print("Data not found! Please run prepare_tinystories_local.py first")
        sys.exit(1)
    
    # Train
    model, best_loss = train(config)
    print(f"\nExperiment completed successfully!")