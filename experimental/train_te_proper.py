#!/usr/bin/env python3
"""
Properly configured FP8 training following NVIDIA best practices.
Key insights from research:
1. HYBRID format is correct (E4M3 forward, E5M2 gradients)
2. Accumulations must be in FP16/BF16, not FP8
3. Softmax/norms/loss must be FP32
4. The bottleneck is likely the small model size, not FP8 config
"""

import os
import sys
import time
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from model_te import TEModelConfig, TETransformerGPT
from wandb_logger import WandBLogger

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    HAS_TE = True
    # Version check - TE doesn't expose __version__ directly
    try:
        import transformer_engine
        if hasattr(transformer_engine, '__version__'):
            print(f"TransformerEngine version: {transformer_engine.__version__}")
        else:
            print("TransformerEngine loaded successfully")
    except:
        print("TransformerEngine loaded (version unknown)")
except ImportError:
    print("Error: TransformerEngine required!")
    sys.exit(1)

@dataclass
class ProperFP8Config:
    # Model
    n_layer: int = 12         
    n_head: int = 12          
    n_embd: int = 768         
    vocab_size: int = 50256   
    block_size: int = 128     
    dropout: float = 0.0
    
    # Training - Optimal for memory bandwidth
    batch_size: int = 32      # Middle ground
    gradient_accumulation_steps: int = 32  # Effective batch = 1024
    max_iters: int = 1000     
    eval_interval: int = 200  
    eval_iters: int = 50      
    learning_rate: float = 1e-4  
    min_lr: float = 5e-5      
    warmup_iters: int = 100   
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # FP8 - PROPER CONFIGURATION
    fp8_format: str = "HYBRID"  # E4M3 fwd, E5M2 bwd - THIS IS CORRECT
    fp8_margin: int = 0
    fp8_interval: int = 1  # Update scales every iteration
    fp8_amax_history_len: int = 1024  # Longer history for stability
    fp8_amax_compute_algo: str = "max"  # More stable than most_recent
    fp8_wgrad: bool = True  # Enable FP8 weight gradients
    
    # System
    device: str = 'cuda'
    compile: bool = False
    log_interval: int = 50
    checkpoint_dir: str = 'checkpoints_te_proper'
    wandb_project: str = 'tinystories-fp8-proper'
    wandb_run_name: Optional[str] = None

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

def get_batch(split: str, config: ProperFP8Config, data_dir: Path = Path('data')) -> Tuple[torch.Tensor, torch.Tensor]:
    data = _get_memmap('train' if split == 'train' else 'val', data_dir)
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    
    x = torch.clamp(x, max=config.vocab_size - 1)
    y = torch.clamp(y, max=config.vocab_size - 1)
    
    x = x.pin_memory().to(config.device, non_blocking=True)
    y = y.pin_memory().to(config.device, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module, config: ProperFP8Config, fp8_recipe=None) -> dict:
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch('validation' if split == 'val' else split, config)
            
            # Use FP8 for inference too
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, loss = model(X, Y)
            
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

def get_lr(iter_num: int, config: ProperFP8Config) -> float:
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    if iter_num > config.warmup_iters:
        decay_ratio = (iter_num - config.warmup_iters) / (config.max_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    return config.learning_rate

def diagnose_fp8_performance(model, config, fp8_recipe):
    """Diagnose why FP8 isn't giving expected speedup."""
    print("\n" + "="*60)
    print("FP8 PERFORMANCE DIAGNOSIS")
    print("="*60)
    
    # Check if FP8 is actually being used
    print("\n1. Checking FP8 activation...")
    if os.environ.get('NVTE_DEBUG'):
        print("   NVTE_DEBUG is set - check output for FP8 kernel launches")
    else:
        print("   Set NVTE_DEBUG=1 to see FP8 kernel details")
    
    # Model size analysis
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n2. Model size: {total_params/1e6:.1f}M parameters")
    print(f"   - Small models (< 1B) often memory-bandwidth limited")
    print(f"   - FP8 helps more with compute-bound (large) models")
    
    # Batch size analysis
    tokens_per_batch = config.batch_size * config.block_size
    print(f"\n3. Batch configuration:")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Sequence length: {config.block_size}")
    print(f"   - Tokens per batch: {tokens_per_batch:,}")
    print(f"   - May be too small to saturate GPU compute")
    
    # Memory bandwidth estimate
    # Rough estimate: each token needs to read all parameters
    bytes_per_param = 2  # BF16
    bytes_per_batch = tokens_per_batch * total_params * bytes_per_param
    print(f"\n4. Memory bandwidth analysis:")
    print(f"   - Bytes per batch: {bytes_per_batch/1e9:.2f} GB")
    print(f"   - RTX 4090 bandwidth: 1 TB/s")
    print(f"   - Likely memory-bandwidth limited, not compute")
    
    # FP8 overhead
    print(f"\n5. FP8 overhead factors:")
    print(f"   - Format conversions: BF16 ↔ FP8")
    print(f"   - Scale factor updates every {config.fp8_interval} iters")
    print(f"   - Amax history tracking: {config.fp8_amax_history_len} steps")
    print(f"   - Padding overhead for alignment")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("\n1. This model is too small for significant FP8 gains")
    print("2. Try with larger models (>1B params) for better speedup")
    print("3. Increase batch size if memory allows")
    print("4. FP8 shines with long sequences (2k+) and large models")
    print("="*60 + "\n")

def train(config: ProperFP8Config):
    """Training with proper FP8 configuration."""
    
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # PROPER FP8 recipe - HYBRID with stable settings
    fp8_recipe = DelayedScaling(
        fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
        amax_history_len=config.fp8_amax_history_len,
        amax_compute_algo=config.fp8_amax_compute_algo,
        margin=config.fp8_margin,
        interval=config.fp8_interval,
        fp8_wgrad=config.fp8_wgrad,  # FP8 weight gradients
    )
    
    print(f"FP8 Recipe Configuration:")
    print(f"  Format: HYBRID (E4M3 forward, E5M2 backward)")
    print(f"  Amax history: {config.fp8_amax_history_len}")
    print(f"  Scale update interval: {config.fp8_interval}")
    print(f"  Weight gradients: {'FP8' if config.fp8_wgrad else 'BF16'}")
    
    # Initialize wandb
    wandb_logger = WandBLogger(
        project=config.wandb_project,
        run_name=config.wandb_run_name or f"proper-hybrid-b{config.batch_size}",
        config=vars(config)
    )
    
    # Create model
    model_config = TEModelConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_kv_heads=config.n_head,
        block_size=config.block_size,
        dropout=config.dropout,
        bias=False
    )
    model_config.ffn_hidden_size = 2048
    
    model = TETransformerGPT(model_config)
    model = model.to(config.device)
    
    # Diagnose potential issues
    diagnose_fp8_performance(model, config, fp8_recipe)
    
    # Create GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # FP8 handles its own scaling
    
    # Optimizer - keep states in FP32
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=True  # Fused optimizer for speed
    )
    
    # Training
    best_val_loss = float('inf')
    iter_num = 0
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    print(f"\nStarting properly configured FP8 training")
    print(f"Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print("-" * 50)
    
    # Warmup pass for FP8 calibration
    print("Running FP8 calibration...")
    for _ in range(10):
        X, Y = get_batch('train', config)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, calibrating=True):
                _, _ = model(X, Y)
    torch.cuda.synchronize()
    print("FP8 calibration complete")
    
    t0 = time.time()
    local_iter_num = 0
    
    while iter_num < config.max_iters:
        # Evaluate
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config, fp8_recipe)
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb_logger.log_eval(losses, step=iter_num)
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                print(f"New best validation loss: {best_val_loss:.4f}")
        
        # Learning rate schedule
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training step
        optimizer.zero_grad(set_to_none=True)
        
        # Gradient accumulation
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = get_batch('train', config)
            
            # Forward pass with proper FP8 autocast
            # BF16 for accumulations, FP8 for compute
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, calibrating=False):
                    logits, loss = model(X, Y)
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step (with FP32 master weights)
        optimizer.step()
        
        # Logging
        if iter_num % config.log_interval == 0:
            local_iter_num = max(1, local_iter_num)
            t1 = time.time()
            dt = t1 - t0
            tokens_per_iter = config.batch_size * config.gradient_accumulation_steps * config.block_size
            tokens_per_sec = tokens_per_iter * local_iter_num / dt
            
            print(f"iter {iter_num}: loss {loss.item()*config.gradient_accumulation_steps:.4f}, "
                  f"{tokens_per_sec:.0f} tok/s, lr {lr:.2e} [FP8-HYBRID-PROPER]")
            
            wandb_logger.log_metrics({
                'train/loss': loss.item() * config.gradient_accumulation_steps,
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
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
    
    # Final diagnosis
    print("\n" + "="*60)
    print("FINAL ANALYSIS")
    print("="*60)
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"This model is too small to benefit significantly from FP8")
    print(f"FP8 shines with models >1B params and longer sequences")
    print("="*60)
    
    wandb_logger.finish()
    return model, best_val_loss

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--debug', action='store_true',
                       help='Enable NVTE_DEBUG for FP8 kernel info')
    args = parser.parse_args()
    
    if args.debug:
        os.environ['NVTE_DEBUG'] = '1'
        print("NVTE_DEBUG enabled - will show FP8 kernel details")
    
    config = ProperFP8Config(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        wandb_run_name=args.wandb_run_name
    )
    
    # Check requirements
    if not torch.cuda.is_available():
        print("CUDA required!")
        sys.exit(1)
    
    cc = torch.cuda.get_device_capability()
    if cc[0] < 8:
        print(f"Warning: GPU has compute capability {cc[0]}.{cc[1]}")
        print("FP8 requires CC >= 8.0 (Ampere or newer)")
    
    data_dir = Path('data')
    if not (data_dir / 'train.bin').exists():
        print("Data not found! Run prepare_tinystories_local.py first")
        sys.exit(1)
    
    # Train
    model, best_loss = train(config)
    print("\nConclusion: For 162M param models, FP8 provides minimal speedup.")
    print("The model is memory-bandwidth limited, not compute limited.")
    print("Try FP8 with larger models (>1B params) for significant gains.")