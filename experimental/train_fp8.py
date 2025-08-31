#!/usr/bin/env python3
"""
FP8 Training script using TransformerEngine.
Optimized for H100/H200 GPUs with native FP8 Tensor Core support.
Note: A100 lacks FP8 Tensor Cores - will run but without speedup.
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

# TransformerEngine imports
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    print("✓ TransformerEngine loaded successfully")
except ImportError:
    print("ERROR: TransformerEngine not found!")
    print("Install with: pip install transformer-engine")
    import sys
    sys.exit(1)

from model_te import ModelConfig, SimpleGPT_TE, get_fp8_recipe, load_from_bfloat16_checkpoint

# Check for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Note: wandb not available, logging disabled")


@dataclass
class TrainingConfig:
    # Model config (matches your 112M model)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 49152  # SmolLM
    n_kv_heads: int = 3      # GQA compression
    block_size: int = 2048
    dropout: float = 0.05
    
    # FP8 config
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_warmup_steps: int = 100  # Run BF16 for initial steps to stabilize
    
    # Training config
    batch_size: int = 16
    gradient_accumulation_steps: int = 16
    max_iters: int = 8000
    eval_interval: int = 100
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    lr_decay_iters: int = 8000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # System
    device: str = 'cuda'
    compile: bool = True  # torch.compile
    compile_mode: str = 'max-autotune'
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = 'checkpoints_fp8'
    data_dir: str = 'data_fineweb_edu_smollm'
    
    # Wandb
    wandb_project: str = 'fp8-training'
    wandb_run_name: Optional[str] = None
    
    # Resume
    resume_from: Optional[str] = None


class DataLoader:
    """Memory-mapped data loader - optimized version."""
    def __init__(self, data_dir, block_size, device='cuda'):
        self.block_size = block_size
        self.device = device
        
        # Load memory-mapped arrays
        data_path = Path(data_dir)
        self.train_data = np.memmap(data_path / 'train.bin', dtype=np.uint16, mode='r')
        self.val_data = np.memmap(data_path / 'val.bin', dtype=np.uint16, mode='r')
        
        print(f"Loaded data from {data_dir}")
        print(f"  Train: {len(self.train_data):,} tokens")
        print(f"  Val: {len(self.val_data):,} tokens")
    
    def get_batch(self, split, batch_size):
        """Get a batch - no padding needed as block_size=2048 is already divisible by 16."""
        data = self.train_data if split == 'train' else self.val_data
        
        # Random starting positions
        max_start = len(data) - self.block_size - 1
        ix = torch.randint(max_start, (batch_size,))
        
        # Vectorized data loading
        offsets = torch.arange(self.block_size + 1)
        indices = ix.unsqueeze(1) + offsets.unsqueeze(0)
        # Pin memory for faster CPU->GPU transfer
        batch_data = torch.from_numpy(data[indices.numpy()].astype(np.int64)).pin_memory()
        
        # Non-blocking transfer for better overlap
        x = batch_data[:, :-1].to(self.device, non_blocking=True)
        y = batch_data[:, 1:].to(self.device, non_blocking=True)
        
        return x, y


def get_lr(iter_num, config):
    """Learning rate schedule with warmup and cosine decay."""
    # Warmup
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    
    # Plateau (important for stability)
    plateau_iters = int(config.lr_decay_iters * 0.6)
    if iter_num < config.warmup_iters + plateau_iters:
        return config.learning_rate
    
    # Cosine decay
    decay_iter = iter_num - config.warmup_iters - plateau_iters
    decay_length = config.lr_decay_iters - config.warmup_iters - plateau_iters
    
    if decay_iter > decay_length:
        return config.min_lr
    
    decay_ratio = decay_iter / decay_length
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def evaluate(model, data_loader, config, fp8_recipe):
    """Evaluate model on validation set."""
    model.eval()
    losses = []
    
    for _ in range(config.eval_iters):
        x, y = data_loader.get_batch('val', config.batch_size)
        
        # Use FP8 for evaluation too
        if config.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def save_checkpoint(model, optimizer, config, iter_num, val_loss, checkpoint_path):
    """Save model checkpoint with FP8 metadata."""
    # Get raw model state dict (unwrap compiled model if needed)
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': asdict(config),
        'iter_num': iter_num,
        'val_loss': val_loss,
        'best_val_loss': val_loss,
    }
    
    # Save FP8 recipe metadata if using FP8
    if config.use_fp8:
        checkpoint['fp8_metadata'] = {
            'amax_history_len': config.fp8_amax_history_len,
            'format': 'HYBRID',  # E4M3 fwd, E5M2 bwd
        }
    
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)


def train():
    """Main training loop with FP8."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from', type=str, help='Resume from checkpoint')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--no_fp8', action='store_true', help='Disable FP8')
    parser.add_argument('--data_dir', type=str, default='data_fineweb_edu_smollm', help='Data directory')
    parser.add_argument('--max_iters', type=int, default=8000, help='Maximum iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    args = parser.parse_args()
    
    # Configuration
    config = TrainingConfig()
    if args.resume_from:
        config.resume_from = args.resume_from
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_fp8:
        config.use_fp8 = False
    config.compile = args.compile
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
    )
    
    # Create model
    model = SimpleGPT_TE(model_config).to(config.device)
    # Cast model to BF16 (crucial for memory/bandwidth)
    model = model.to(torch.bfloat16)
    
    # Get FP8 recipe (auto-detects Blackwell and uses MXFP8 if available)
    fp8_recipe = get_fp8_recipe(model_config, use_mx=None)  # Auto-detect GPU type
    
    # Initialize or resume
    iter_num = 0
    best_val_loss = float('inf')
    last_val_loss = float('inf')  # Track for checkpointing
    
    if config.resume_from:
        if config.resume_from.endswith('.pt'):
            # Loading from BF16 checkpoint
            iter_num, best_val_loss = load_from_bfloat16_checkpoint(model, config.resume_from)
            print(f"Resumed from BF16 checkpoint at iteration {iter_num}")
        else:
            # Loading from FP8 checkpoint
            try:
                checkpoint = torch.load(config.resume_from, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(config.resume_from, map_location='cpu')
            
            state_dict = checkpoint['model']
            # Handle compiled model prefix if present
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                print("Removing _orig_mod prefix from checkpoint")
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            iter_num = checkpoint.get('iter_num', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from FP8 checkpoint at iteration {iter_num}")
    
    # Compile model if requested
    if config.compile:
        print(f"Compiling model...")
        # Use default mode with options to disable CUDA graphs (fixes TE warnings)
        # Can't use both mode='max-autotune' and options together
        model = torch.compile(model, mode='default',
                            options={"triton.cudagraphs": False})
    
    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=True  # Use fused AdamW on CUDA
    )
    
    # Data loader
    data_loader = DataLoader(config.data_dir, config.block_size, config.device)
    
    # Wandb logging
    if WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"fp8_{config.n_layer}L_{config.n_embd}D",
            config=asdict(config)
        )
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting FP8 Training")
    print("="*50)
    print(f"Model: {model_config.n_layer}L, {model_config.n_head}H, {model_config.n_embd}D")
    print(f"Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"FP8 enabled: {config.use_fp8}")
    if config.use_fp8:
        print("  WARNING: A100 lacks FP8 Tensor Cores. Use H100/H200 for speedup.")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print("="*50 + "\n")
    
    model.train()
    optimizer.zero_grad()
    
    t0 = time.time()
    tokens_processed = 0
    
    for iter_num in range(iter_num, config.max_iters):
        # Learning rate schedule
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Accumulate gradients
        total_loss = 0
        
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = data_loader.get_batch('train', config.batch_size)
            
            # Decide whether to use FP8 (after warmup)
            use_fp8_now = config.use_fp8 and (iter_num >= config.fp8_warmup_steps)
            
            # Forward pass with FP8 autocast
            if use_fp8_now:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, loss = model(x, y)
            else:
                # BF16 forward pass (during warmup)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            
            # Store unscaled loss for accurate logging
            total_loss += loss.item()
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            
            # Backward pass (outside FP8 autocast)
            loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Update token count
        tokens_processed += config.batch_size * config.block_size * config.gradient_accumulation_steps
        
        # Logging
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = tokens_processed / dt if dt > 0 else 0
            
            # Note: total_loss is the sum of unscaled losses, already averaged over accumulation steps
            avg_loss = total_loss / config.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {avg_loss:.4f}, lr {lr:.2e}, "
                  f"{tokens_per_sec/1e3:.1f}k tok/s, FP8: {use_fp8_now}")
            
            if WANDB_AVAILABLE:
                wandb.log({
                    'train/loss': total_loss / config.gradient_accumulation_steps,
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
            last_val_loss = val_loss  # Update last_val_loss
            print(f"Step {iter_num}: val loss {val_loss:.4f}")
            
            if WANDB_AVAILABLE:
                wandb.log({'val/loss': val_loss}, step=iter_num)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, config, iter_num, val_loss,
                    Path(config.checkpoint_dir) / 'best_model_fp8.pt'
                )
        
        # Regular checkpoints (use best_val_loss if no eval ran yet)
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            checkpoint_val_loss = last_val_loss if last_val_loss != float('inf') else best_val_loss
            save_checkpoint(
                model, optimizer, config, iter_num, checkpoint_val_loss,
                Path(config.checkpoint_dir) / f'checkpoint_{iter_num}_fp8.pt'
            )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*50)


if __name__ == "__main__":
    train()