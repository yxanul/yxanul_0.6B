#!/usr/bin/env python3
"""
Fair comparison training script - Same 113M model size for FP8 vs BF16.
Uses minimal padding to maintain model size while meeting FP8 requirements.
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
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
except ImportError:
    print("Error: TransformerEngine required!")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration - Matching original BF16 model
@dataclass
class TrainingConfig:
    # Model - Same as BF16 baseline (113M params)
    n_layer: int = 12         
    n_head: int = 12          
    n_embd: int = 768         
    vocab_size: int = 50256   # Padded to multiple of 16 for FP8 (50256 ÷ 16 = 3141)
    block_size: int = 128     # Same as original
    dropout: float = 0.0
    
    # Training - Same as BF16 baseline
    batch_size: int = 64      # Same as original
    gradient_accumulation_steps: int = 16  # Same as original
    max_iters: int = 1000     
    eval_interval: int = 200  
    eval_iters: int = 100      
    learning_rate: float = 1e-4  
    min_lr: float = 5e-5      
    warmup_iters: int = 1000  # Proper warmup
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # FP8 specific - HYBRID format
    fp8_format: str = "HYBRID"  
    fp8_margin: float = 0
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    fp8_calibration_steps: int = 10
    force_bf16: bool = False  # Option to force BF16 for comparison
    
    # System
    device: str = 'cuda'
    compile: bool = False
    
    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 5000
    checkpoint_dir: str = 'checkpoints_te_fair'
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
    
    # Clamp to padded vocab size
    x = torch.clamp(x, max=config.vocab_size - 1)
    y = torch.clamp(y, max=config.vocab_size - 1)
    
    x = x.pin_memory().to(config.device, non_blocking=True)
    y = y.pin_memory().to(config.device, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module, config: TrainingConfig, fp8_recipe=None, use_fp8=False) -> dict:
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch('validation' if split == 'val' else split, config)
            
            if use_fp8 and fp8_recipe is not None:
                # FP8 evaluation
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        logits, loss = model(X, Y)
            else:
                # BF16 evaluation
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
    """Main training loop - fair comparison."""
    
    # Seeds & optimizations
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Determine precision mode
    use_fp8 = not config.force_bf16
    precision_name = "BF16" if config.force_bf16 else "FP8-HYBRID"
    
    # Create FP8 recipe if needed
    fp8_recipe = None
    if use_fp8:
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=config.fp8_amax_history_len,
            amax_compute_algo=config.fp8_amax_compute_algo,
        )
        print(f"FP8 Recipe: HYBRID format")
    
    # Initialize wandb
    wandb_logger = WandBLogger(
        project=config.wandb_project,
        run_name=config.wandb_run_name or f"tinystories-te-fair-{precision_name.lower()}",
        config=vars(config)
    )
    
    # Create model - override FFN size to match original
    model_config = TEModelConfig(
        vocab_size=config.vocab_size,  # 50272 (divisible by 16)
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_kv_heads=config.n_head,
        block_size=config.block_size,
        dropout=config.dropout,
        bias=False
    )
    # Override FFN size to match original model (not the auto-calculated one)
    # Original uses 768 * 8/3 = 2048, but we want smaller for 113M total
    model_config.ffn_hidden_size = 2048  # This gives us ~113M params
    
    model = TETransformerGPT(model_config)
    model = model.to(config.device)
    
    print(f"\nModel configuration (Fair Comparison):")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Vocab size: {config.vocab_size} (padded by {config.vocab_size - 50257})")
    print(f"  Block size: {config.block_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Precision: {precision_name}")
    
    # GPU info
    cc = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name} (CC {cc[0]}.{cc[1]})")
    
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
    
    print(f"\nStarting {precision_name} training (Fair Comparison)")
    print(f"Iterations: {config.max_iters}")
    print(f"Tokens per iteration: {config.batch_size * config.gradient_accumulation_steps * config.block_size:,}")
    print("-" * 50)
    
    # Test FP8 if enabled
    if use_fp8:
        try:
            with torch.no_grad():
                Xd = torch.randint(0, config.vocab_size, (2, 64), device=config.device)
                Yd = Xd.clone()
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, calibrating=True):
                        _ = model(Xd, Yd)
                torch.cuda.synchronize()
            print(f"{precision_name} preflight test passed")
        except RuntimeError as e:
            print(f"{precision_name} not supported, falling back to BF16")
            use_fp8 = False
            precision_name = "BF16"
    
    t0 = time.time()
    local_iter_num = 0
    
    while iter_num < config.max_iters:
        # Determine if calibrating (FP8 only)
        calibrating = use_fp8 and (iter_num < config.fp8_calibration_steps)
        
        # Evaluate periodically
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config, fp8_recipe, use_fp8)
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb_logger.log_eval(losses, step=iter_num)
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'precision': precision_name
                }
                torch.save(checkpoint, Path(config.checkpoint_dir) / f'best_model_{precision_name.lower()}.pt')
                print(f"Saved new best model (val loss: {best_val_loss:.4f})")
        
        # Adjust learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training step
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = get_batch('train', config)
            
            # Forward pass
            if use_fp8:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, calibrating=calibrating):
                        logits, loss = model(X, Y)
            else:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
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
            
            status = f"[{precision_name}]"
            if use_fp8 and calibrating:
                status += " (calibrating)"
            
            print(f"iter {iter_num}: loss {loss.item()*config.gradient_accumulation_steps:.4f}, "
                  f"{tokens_per_sec:.0f} tok/s, lr {lr:.2e} {status}")
            
            wandb_logger.log_metrics({
                'train/loss': loss.item() * config.gradient_accumulation_steps,
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
                'train/precision': precision_name,
            }, step=iter_num)
            
            t0 = time.time()
            local_iter_num = 0
        
        iter_num += 1
        local_iter_num += 1
    
    # Final evaluation
    print("\nTraining complete!")
    losses = estimate_loss(model, config, fp8_recipe, use_fp8)
    print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Perplexity: {math.exp(best_val_loss):.2f}")
    
    # Summary statistics
    print(f"\n{'='*50}")
    print(f"Summary for {precision_name}:")
    print(f"  Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"  Best loss: {best_val_loss:.4f}")
    print(f"  Perplexity: {math.exp(best_val_loss):.2f}")
    print(f"  Precision: {precision_name}")
    print(f"{'='*50}")
    
    wandb_logger.set_summary(
        best_val_loss=best_val_loss,
        final_train_loss=losses['train'],
        final_val_loss=losses['val'],
        perplexity=math.exp(best_val_loss),
        model_size_m=sum(p.numel() for p in model.parameters())/1e6,
        precision=precision_name
    )
    wandb_logger.finish()
    
    return model, best_val_loss

# -----------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--force_bf16', action='store_true',
                       help='Force BF16 mode for comparison')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()
    
    config = TrainingConfig(
        max_iters=args.max_iters,
        force_bf16=args.force_bf16,
        wandb_run_name=args.wandb_run_name
    )
    
    # Check requirements
    if not torch.cuda.is_available():
        print("CUDA required!")
        sys.exit(1)
    
    data_dir = Path('data')
    if not (data_dir / 'train.bin').exists():
        print("Data not found! Run prepare_tinystories_local.py first")
        sys.exit(1)
    
    # Train
    model, best_loss = train(config)
    print("\nFair comparison experiment complete!")