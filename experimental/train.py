#!/usr/bin/env python3
"""
Simple, clean training script.
No complications - just solid training with good monitoring.
"""

import os
import time
import math
import json
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import create_model, ModelConfig
from wandb_logger import WandBLogger


# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------

# Model
vocab_size = 50257  # GPT-2 vocab size (or set from metadata.json)

# Data
data_dir = "data"  # Relative to experimental/ folder
batch_size = 4  # Micro batch size
block_size = 2048  # Context length
gradient_accumulation_steps = 32  # Effective batch = 4 * 32 = 128

# Training
max_iters = 100000
warmup_iters = 2000
learning_rate = 3e-4
min_lr = 3e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.98  # More stable than 0.999 for larger models
grad_clip = 1.0

# Evaluation
eval_interval = 500
log_interval = 10  # Log more frequently to see progress
eval_iters = 20    # Much faster evaluation (was 200!)

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
compile_model = False  # torch.compile (set True if you have PyTorch 2.0+)

# Output
out_dir = "outputs"  # Relative to experimental/ folder
save_interval = 5000

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load metadata if exists
metadata_path = Path(data_dir) / "metadata.json"
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        vocab_size = metadata['vocab_size']
        print(f"Loaded metadata: vocab_size={vocab_size}")

# Auto-detect dtype and context
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Prepare training configuration payload for logging
train_hparams = {
    'vocab_size': vocab_size,
    'data_dir': data_dir,
    'batch_size': batch_size,
    'block_size': block_size,
    'gradient_accumulation_steps': gradient_accumulation_steps,
    'max_iters': max_iters,
    'warmup_iters': warmup_iters,
    'learning_rate': learning_rate,
    'min_lr': min_lr,
    'weight_decay': weight_decay,
    'beta1': beta1,
    'beta2': beta2,
    'grad_clip': grad_clip,
    'eval_interval': eval_interval,
    'eval_iters': eval_iters,
    'device': device,
    'dtype': dtype,
    'compile_model': compile_model,
}

# Data loader
def get_batch(split):
    """Load a batch of data."""
    # Memory map the data file
    data_path = Path(data_dir) / f"{split}.bin"
    
    # Detect dtype from file size and vocab
    if vocab_size > 65535:
        data_dtype = np.uint32
    else:
        data_dtype = np.uint16
    
    data = np.memmap(data_path, dtype=data_dtype, mode='r')
    
    # Generate random positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Load data
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Move to device
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y


# Model initialization
print(f"Initializing model with vocab_size={vocab_size}")
model, config = create_model(vocab_size=vocab_size)
model = model.to(device)

# WandB logger
wandb_logger = WandBLogger(
    enabled=os.getenv('WANDB_DISABLED', 'false').lower() not in ('true', '1', 'yes'),
    project=os.getenv('WANDB_PROJECT', 'yxanul'),
    run_name=os.getenv('WANDB_RUN_NAME'),
    config={
        **train_hparams,
        **config.__dict__,
        'num_parameters': getattr(model, 'num_parameters', lambda: sum(p.numel() for p in model.parameters() if p.requires_grad))(),
    },
    entity=os.getenv('WANDB_ENTITY'),
    mode=os.getenv('WANDB_MODE'),
)
wandb_logger.watch(model, log='gradients', log_freq=max(1, eval_interval // 2))

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
    eps=1e-8
)

# Compile model if requested (PyTorch 2.0+)
if compile_model:
    print("Compiling model... (this may take a minute)")
    model = torch.compile(model)

# Learning rate scheduler
def get_lr(it):
    """Learning rate schedule with warmup and cosine decay."""
    # Warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay
    if it > max_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)


# Evaluation
@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and val sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Training loop
def train():
    """Main training loop."""
    
    # Training variables
    best_val_loss = 1e9
    iter_num = 0
    
    # Timing
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    print(f"Starting training...")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Max iterations: {max_iters}")
    print("-" * 50)
    
    X, Y = get_batch('train')  # Fetch first batch
    
    while True:
        # Determine learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate (skip first evaluation for faster startup)
        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss()
            train_loss = float(losses['train'])
            val_loss = float(losses['val'])
            print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            # Log eval metrics + perplexity
            wandb_logger.log_metrics({
                'eval/train_loss': train_loss,
                'eval/train_ppl': math.exp(min(20.0, train_loss)),
                'eval/val_loss': val_loss,
                'eval/val_ppl': math.exp(min(20.0, val_loss)),
            }, step=iter_num)
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': config.__dict__,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
        # Save periodic checkpoint
        if iter_num > 0 and iter_num % save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': config.__dict__,
                'iter_num': iter_num,
                'val_loss': losses.get('val', 0),
            }
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
        
        # Forward backward update, with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps  # Scale loss
            
            # Get next batch asynchronously
            X, Y = get_batch('train')
            
            # Backward
            loss.backward()
        
        # Clip gradients
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0:
            # Calculate tokens per second
            tokens_per_sec = batch_size * block_size * gradient_accumulation_steps / dt
            
            # Get loss value for printing
            lossf = loss.item() * gradient_accumulation_steps
            
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, tokens/sec {tokens_per_sec:.0f}, lr {lr:.2e}")
            # Log training metrics
            wandb_logger.log_metrics({
                'train/loss': lossf,
                'train/ppl': math.exp(min(20.0, lossf)),
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
                'train/iter_time_ms': dt * 1000.0,
                'train/iter': iter_num,
            }, step=iter_num)
        
        iter_num += 1
        local_iter_num += 1
        
        # Termination
        if iter_num > max_iters:
            break


# Simple test before training
def test_model():
    """Test model before training."""
    print("Testing model...")
    
    # Test data loading
    X, Y = get_batch('train')
    print(f"Data batch shape: X={X.shape}, Y={Y.shape}")
    
    # Test forward pass
    with torch.no_grad():
        with ctx:
            logits, loss = model(X, Y)
    print(f"Initial loss: {loss.item():.4f}")
    
    # Test memory
    if device_type == 'cuda':
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    print("Model test passed!")
    print("-" * 50)


if __name__ == "__main__":
    # Run test
    test_model()
    
    # Start training
    train()
    # Finish WandB run
    wandb_logger.finish()
    
    print("\nTraining complete!")
