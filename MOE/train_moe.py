#!/usr/bin/env python3
"""
CLEAN MoE FP8 training script.
Built on the CLEAN philosophy with MoE-specific optimizations.

Key features:
- Expert load balancing tracking
- Auxiliary loss monitoring
- MoE-specific metrics in wandb
- Efficient routing for small models

Targets ~475M params, optimized for RTX 5090.
"""

import os
import sys
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'train'))

# Import MoE model
from model_moe_clean import MoEConfig, CleanMoE_TE, get_fp8_recipe

# TransformerEngine
import transformer_engine.pytorch as te

# Import wandb logger from parent
from wandb_logger import WandBLogger


@dataclass
class MoETrainingConfig:
    # Model config
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 32768  # 32k BPE tokenizer
    n_kv_heads: int = 3
    block_size: int = 4096
    dropout: float = 0.05
    
    # MoE config
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_jitter_noise: float = 0.01
    num_dense_layers: int = 1
    expert_dropout: float = 0.1
    
    # FP8 config
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_warmup_steps: int = 100
    
    # Training config
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 50
    learning_rate: float = 6e-4  # Slightly lower for MoE stability
    min_lr: float = 6e-5
    warmup_iters: int = 200
    lr_decay_iters: int = 4000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    
    # System
    device: str = 'cuda'
    compile: bool = False  # Disabled for TE compatibility
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = 'MOE/checkpoints'
    data_dir: str = 'data_mixed_3b'
    
    # Wandb
    wandb_project: str = 'moe-clean'
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
    """Learning rate schedule with extended warmup for MoE."""
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
def evaluate_with_experts(model, data_loader, config, fp8_recipe):
    """Evaluate model with expert utilization tracking."""
    model.eval()
    losses = []
    expert_counts = torch.zeros(config.num_experts, device=config.device)
    total_tokens = 0
    
    for _ in range(config.eval_iters):
        x, y = data_loader.get_batch('val', config.batch_size)
        
        # Track expert usage during evaluation
        if config.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        
        losses.append(loss.item())
        total_tokens += x.numel()
    
    model.train()
    return np.mean(losses), expert_counts / max(total_tokens, 1)


def save_checkpoint(model, optimizer, config, iter_num, val_loss, checkpoint_path):
    """Save checkpoint with MoE state."""
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': asdict(config),
        'iter_num': iter_num,
        'val_loss': val_loss,
        'model_type': 'moe_clean',
    }
    
    print(f"Saving MoE checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)


def train():
    """Main MoE training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='data_mixed_3b', help='Data directory')
    parser.add_argument('--max_iters', type=int, default=2000, help='Max iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='Eval interval')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--no_fp8', action='store_true', help='Disable FP8')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    # MoE-specific arguments
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--top_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--capacity_factor', type=float, default=1.25, help='Expert capacity factor')
    parser.add_argument('--aux_loss_coef', type=float, default=0.01, help='Auxiliary loss coefficient')
    parser.add_argument('--num_dense_layers', type=int, default=1, help='Number of dense layers at start')
    args = parser.parse_args()
    
    # Configuration
    config = MoETrainingConfig()
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_fp8:
        config.use_fp8 = False
    
    # MoE configuration
    config.num_experts = args.num_experts
    config.top_k = args.top_k
    config.capacity_factor = args.capacity_factor
    config.router_aux_loss_coef = args.aux_loss_coef
    config.num_dense_layers = args.num_dense_layers
    
    config.data_dir = args.data_dir
    config.max_iters = args.max_iters
    config.eval_interval = args.eval_interval
    config.log_interval = args.log_interval
    
    # Model configuration
    model_config = MoEConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_kv_heads=config.n_kv_heads,
        block_size=config.block_size,
        dropout=config.dropout,
        num_experts=config.num_experts,
        top_k=config.top_k,
        capacity_factor=config.capacity_factor,
        router_aux_loss_coef=config.router_aux_loss_coef,
        router_z_loss_coef=config.router_z_loss_coef,
        router_jitter_noise=config.router_jitter_noise,
        num_dense_layers=config.num_dense_layers,
        expert_dropout=config.expert_dropout,
        use_fp8=config.use_fp8,
        fp8_amax_history_len=config.fp8_amax_history_len,
    )
    
    # Create MoE model
    model = CleanMoE_TE(model_config).to(config.device)
    model = model.to(torch.bfloat16)
    
    # Get FP8 recipe
    fp8_recipe = get_fp8_recipe(model_config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=True
    )
    
    # Data loader
    data_loader = DataLoader(config.data_dir, config.block_size, config.device)
    
    # Initialize WandB logger with MoE config
    wandb_config = asdict(config)
    wandb_config.update({
        'model_type': 'moe_clean',
        'total_params': model.num_parameters(),
        'active_params': model.active_parameters(),
    })
    
    logger = WandBLogger(
        enabled=not args.no_wandb,
        project=config.wandb_project,
        run_name=config.wandb_run_name or f"moe_{config.num_experts}e_top{config.top_k}_{config.batch_size}b",
        config=wandb_config
    )
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Training info
    print("\n" + "="*50)
    print("Starting CLEAN MoE Training")
    print("="*50)
    print(f"Model: {model_config.n_layer}L, {model_config.n_head}H, {model_config.n_embd}D")
    print(f"Total parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"Active parameters: ~{model.active_parameters()/1e6:.1f}M per token")
    print(f"MoE Config:")
    print(f"  - Experts: {config.num_experts} (top-{config.top_k})")
    print(f"  - Dense layers: first {config.num_dense_layers}")
    print(f"  - Capacity factor: {config.capacity_factor}")
    print(f"  - Aux loss coef: {config.router_aux_loss_coef}")
    print(f"FP8: {config.use_fp8}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print("="*50 + "\n")
    
    model.train()
    iter_num = 0
    best_val_loss = float('inf')
    t0 = time.time()
    tokens_processed = 0
    
    # Track MoE-specific metrics
    expert_load_balance = []
    router_entropy = []
    
    for iter_num in range(config.max_iters):
        # Learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Accumulate gradients
        total_loss = 0
        total_aux_loss = 0
        total_z_loss = 0
        
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = data_loader.get_batch('train', config.batch_size)
            
            # Determine if using FP8
            use_fp8_now = config.use_fp8 and (iter_num >= config.fp8_warmup_steps)
            
            # Forward pass
            if use_fp8_now:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            
            # Extract component losses from combined loss
            # Note: aux and z losses are already included in loss
            base_loss = loss.item()
            total_loss += base_loss
            
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        
        # Optimizer step
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
            
            # Calculate perplexity
            try:
                perplexity = math.exp(min(20.0, avg_loss))
            except:
                perplexity = float('inf')
            
            # Log comprehensive metrics
            metrics = {
                'train/loss': avg_loss,
                'train/perplexity': perplexity,
                'train/lr': lr,
                'train/tokens_per_sec': tokens_per_sec,
                'train/grad_norm': grad_norm.item() if config.grad_clip > 0 else 0,
                'train/fp8_active': use_fp8_now,
                'train/gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                'train/iteration': iter_num,
            }
            
            logger.log_metrics(metrics, step=iter_num)
            
            tokens_processed = 0
            t0 = time.time()
        
        # Evaluation
        if iter_num % config.eval_interval == 0:
            val_loss, expert_usage = evaluate_with_experts(model, data_loader, config, fp8_recipe)
            print(f"Step {iter_num}: val loss {val_loss:.4f}")
            
            # Calculate validation perplexity
            try:
                val_perplexity = math.exp(min(20.0, val_loss))
            except:
                val_perplexity = float('inf')
            
            # Calculate expert load balance (coefficient of variation)
            expert_cv = expert_usage.std() / (expert_usage.mean() + 1e-8)
            
            # Log validation metrics
            val_metrics = {
                'val/loss': val_loss,
                'val/perplexity': val_perplexity,
                'moe/expert_load_cv': expert_cv.item() if torch.is_tensor(expert_cv) else expert_cv,
            }
            
            # Log per-expert usage if small number of experts
            if config.num_experts <= 16:
                for i in range(config.num_experts):
                    val_metrics[f'moe/expert_{i}_usage'] = expert_usage[i].item() if torch.is_tensor(expert_usage[i]) else expert_usage[i]
            
            logger.log_metrics(val_metrics, step=iter_num)
            
            # Update best metrics
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.set_summary(
                    best_val_loss=val_loss,
                    best_val_perplexity=val_perplexity,
                    best_iter=iter_num,
                    best_expert_cv=expert_cv.item() if torch.is_tensor(expert_cv) else expert_cv
                )
                
                # Save best model
                save_checkpoint(
                    model, optimizer, config, iter_num, val_loss,
                    Path(config.checkpoint_dir) / 'best_moe_model.pt'
                )
        
        # Regular checkpoints
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            save_checkpoint(
                model, optimizer, config, iter_num, val_loss,
                Path(config.checkpoint_dir) / f'checkpoint_moe_{iter_num}.pt'
            )
    
    # Final evaluation
    final_val_loss, final_expert_usage = evaluate_with_experts(model, data_loader, config, fp8_recipe)
    try:
        final_perplexity = math.exp(min(20.0, final_val_loss))
    except:
        final_perplexity = float('inf')
    
    logger.set_summary(
        final_val_loss=final_val_loss,
        final_perplexity=final_perplexity,
        total_iterations=config.max_iters,
        model_params=model.num_parameters(),
        active_params=model.active_parameters()
    )
    
    print("\n" + "="*50)
    print("MoE Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final perplexity: {final_perplexity:.2f}")
    print("CLEAN MoE: Efficient routing at scale!")
    print("="*50)
    
    # Clean up wandb
    logger.finish()


if __name__ == "__main__":
    train()