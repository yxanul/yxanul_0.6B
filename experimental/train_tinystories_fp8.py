#!/usr/bin/env python3
"""
TinyStories training with FP8 mixed precision support.
Requires TransformerEngine for FP8 operations.
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

# Try to import TransformerEngine for FP8
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    HAS_TE = True
    print("TransformerEngine detected - FP8 training available")
except ImportError:
    HAS_TE = False
    print("TransformerEngine not found - falling back to BF16")

# -----------------------------------------------------------------------------
# Configuration
@dataclass
class TrainingConfig:
    # Model - GPT-2 Small (124M parameters)
    n_layer: int = 12         # GPT-2 small
    n_head: int = 12          # GPT-2 small
    n_embd: int = 768         # GPT-2 small
    vocab_size: int = 50257   # GPT-2 vocab size
    block_size: int = 128     # Sequence length
    dropout: float = 0.0
    
    # Training - Optimized for RTX 5090
    batch_size: int = 64      # Increased for RTX 5090
    gradient_accumulation_steps: int = 16  # Reduced since we increased batch_size
    max_iters: int = 1000     # Quick experiments
    eval_interval: int = 200  # More frequent eval for better tracking   
    eval_iters: int = 100      
    learning_rate: float = 1e-4  
    min_lr: float = 5e-5      
    warmup_iters: int = 100    # Faster warmup for FP8 calibration
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # FP8 specific
    use_fp8: bool = True
    fp8_margin: float = 0.0  # Margin for FP8 dynamic scaling
    fp8_interval: int = 1    # Update FP8 scales every N steps
    fp8_format: str = "E4M3"  # or "E5M2" 
    fp8_calibration_steps: int = 10  # Steps to calibrate FP8 scales
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'fp8' if HAS_TE else 'bfloat16'  
    compile: bool = False     # Torch compile disabled for FP8
    
    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 5000
    checkpoint_dir: str = 'checkpoints_tinystories_fp8'
    wandb_project: str = 'tinystories-precision-test'
    wandb_run_name: Optional[str] = None

# -----------------------------------------------------------------------------
# FP8 Model Wrapper
class FP8Model(nn.Module):
    """Wrapper to add FP8 support to existing model."""
    def __init__(self, base_model: nn.Module, config: TrainingConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.use_fp8 = config.use_fp8 and HAS_TE
        
        if self.use_fp8:
            # Create FP8 recipe for dynamic scaling
            self.fp8_recipe = recipe.DelayedScaling(
                margin=config.fp8_margin,
                interval=config.fp8_interval,
                fp8_format=recipe.Format.E4M3 if config.fp8_format == "E4M3" else recipe.Format.E5M2,
            )
            
            # Convert linear layers to FP8
            self._convert_to_fp8()
    
    def _convert_to_fp8(self):
        """Convert Linear layers to TransformerEngine FP8 layers."""
        def replace_linear(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Replace with TE Linear for FP8
                    te_linear = te.Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                    )
                    # Copy weights
                    with torch.no_grad():
                        te_linear.weight.copy_(child.weight)
                        if child.bias is not None:
                            te_linear.bias.copy_(child.bias)
                    setattr(module, name, te_linear)
                else:
                    replace_linear(child)
        
        replace_linear(self.base_model)
        print(f"Converted model to FP8 with {self.config.fp8_format} format")
    
    def forward(self, *args, **kwargs):
        if self.use_fp8:
            with te.fp8_autocast(
                enabled=True,
                calibrating=(self.training and self._step < self.config.fp8_calibration_steps),
                fp8_recipe=self.fp8_recipe,
            ):
                return self.base_model(*args, **kwargs)
        else:
            return self.base_model(*args, **kwargs)
    
    def set_step(self, step: int):
        """Update current training step for FP8 calibration."""
        self._step = step

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
    """Main training loop with FP8 support."""
    
    # Seeds & optimizations
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Initialize wandb
    run_name = config.wandb_run_name or f"tinystories-{config.dtype}"
    if config.use_fp8 and HAS_TE:
        run_name += f"-{config.fp8_format}"
    
    wandb_logger = WandBLogger(
        project=config.wandb_project,
        run_name=run_name,
        config=vars(config)
    )
    
    # Create base model
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
    
    base_model = GPT(model_config)
    
    # Wrap with FP8 if available
    if config.use_fp8 and HAS_TE:
        model = FP8Model(base_model, config)
        model._step = 0
        print("Using FP8 mixed precision training")
    else:
        model = base_model
        print("Using BF16 mixed precision training")
    
    model = model.to(config.device)
    print(f"Created model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
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
    
    print(f"\nStarting {'FP8' if config.use_fp8 and HAS_TE else 'BF16'} training for {config.max_iters} iterations")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Tokens per iteration: {config.batch_size * config.gradient_accumulation_steps * config.block_size:,}")
    print("-" * 50)
    
    t0 = time.time()
    local_iter_num = 0
    
    while iter_num < config.max_iters:
        # Update FP8 step counter
        if hasattr(model, 'set_step'):
            model.set_step(iter_num)
        
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
        
        # Accumulate gradients
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = get_batch('train', config)
            
            # Mixed precision context
            if config.use_fp8 and HAS_TE:
                # FP8 autocast is handled in the model wrapper
                logits, loss = model(X, Y)
            else:
                # BF16 autocast
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
            
            precision = "FP8" if config.use_fp8 and HAS_TE else "BF16"
            print(f"iter {iter_num}: loss {loss.item()*config.gradient_accumulation_steps:.4f}, "
                  f"{tokens_per_sec:.0f} tok/s, lr {lr:.2e} [{precision}]")
            
            wandb_logger.log_metrics({
                'train/loss': loss.item() * config.gradient_accumulation_steps,
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
                'train/precision': precision,
            }, step=iter_num)
            
            t0 = time.time()
            local_iter_num = 0
        
        iter_num += 1
        local_iter_num += 1
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final evaluation
    losses = estimate_loss(model, config)
    print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    wandb_logger.set_summary(
        best_val_loss=best_val_loss,
        final_train_loss=losses['train'],
        final_val_loss=losses['val'],
        precision=precision,
        avg_tokens_per_sec=tokens_per_sec
    )
    wandb_logger.finish()
    
    return model, best_val_loss

# -----------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fp8', action='store_true', default=True,
                       help='Use FP8 mixed precision if available')
    parser.add_argument('--fp8_format', type=str, default='E4M3',
                       choices=['E4M3', 'E5M2'],
                       help='FP8 format (E4M3 for compute, E5M2 for gradients)')
    parser.add_argument('--max_iters', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Custom wandb run name')
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        use_fp8=args.use_fp8,
        fp8_format=args.fp8_format,
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
    print(f"\nTraining completed with best loss: {best_loss:.4f}")
    print(f"Perplexity: {math.exp(best_loss):.2f}")