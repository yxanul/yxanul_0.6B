#!/usr/bin/env python3
"""
SFT Training Script for Elite 2.218 Model
Continues training from the pretrained checkpoint on instruction data.

Aggressive SFT approach - treating as continued pretraining with instruction data.
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

# Import CLEAN model (same architecture as pretraining)
from model_te_clean import ModelConfig, CleanGPT_TE, get_fp8_recipe

# TransformerEngine
import transformer_engine.pytorch as te

# Import our robust wandb logger
from wandb_logger import WandBLogger


@dataclass
class SFTConfig:
    # Model config (inherited from pretrained model)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 49152
    n_kv_heads: int = 3
    block_size: int = 2048
    dropout: float = 0.05
    
    # FP8 config (same as pretraining)
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_warmup_steps: int = 50  # Shorter warmup for SFT
    
    # SFT Training config (full epoch approach with frequent updates)
    batch_size: int = 8  # Small batch for frequent updates
    gradient_accumulation_steps: int = 1  # No accumulation - update every batch
    max_iters: int = 800  # ~1 full epoch (772 actual)
    eval_interval: int = 50  # Eval every ~6% of epoch
    eval_iters: int = 20
    learning_rate: float = 7e-5  # Balanced SFT learning rate (sweet spot)
    min_lr: float = 5e-5  # Lower min LR to match reduced starting LR
    warmup_iters: int = 20  # Slightly longer warmup for 800 iterations
    lr_decay_iters: int = 800  # Decay over the full epoch
    grad_clip: float = 1.0
    weight_decay: float = 0.05  # Reduced weight decay for SFT
    beta1: float = 0.9
    beta2: float = 0.999
    
    # System
    device: str = 'cuda'
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = 'checkpoints_sft'
    data_dir: str = 'data_sft'
    base_model: str = 'best_model_fp8_optimized.pt'
    
    # Wandb
    wandb_project: str = 'sft-elite-model'
    wandb_run_name: Optional[str] = None


class SFTDataLoader:
    """Data loader for SFT with label masking."""
    def __init__(self, data_dir, block_size, device='cuda'):
        self.block_size = block_size
        self.device = device
        
        data_path = Path(data_dir)
        
        # Load tokenized data
        self.train_tokens = np.memmap(data_path / 'train_tokens.bin', dtype=np.uint16, mode='r')
        self.train_labels = np.memmap(data_path / 'train_labels.bin', dtype=np.int32, mode='r')
        self.val_tokens = np.memmap(data_path / 'val_tokens.bin', dtype=np.uint16, mode='r')
        self.val_labels = np.memmap(data_path / 'val_labels.bin', dtype=np.int32, mode='r')
        
        # Load metadata
        with open(data_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded SFT data from {data_dir}")
        print(f"  Train: {len(self.train_tokens):,} tokens, {self.metadata['train_examples']} examples")
        print(f"  Val: {len(self.val_tokens):,} tokens, {self.metadata['val_examples']} examples")
        print(f"  Format: {self.metadata['format']}")
    
    def get_batch(self, split, batch_size):
        """Get a batch with proper label masking for SFT."""
        if split == 'train':
            tokens = self.train_tokens
            labels = self.train_labels
        else:
            tokens = self.val_tokens
            labels = self.val_labels
        
        # Ensure we have enough data
        max_start = len(tokens) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(f"Dataset too small for block_size {self.block_size}")
        
        # Random starting positions
        ix = torch.randint(max_start, (batch_size,))
        
        # Get sequences
        x_data = []
        y_data = []
        
        for i in range(batch_size):
            start_idx = ix[i].item()
            end_idx = start_idx + self.block_size
            
            # Input tokens (shifted by 1 for next-token prediction)
            x_seq = tokens[start_idx:end_idx]
            # Labels (what we want to predict)
            y_seq = labels[start_idx+1:end_idx+1] 
            
            x_data.append(x_seq)
            y_data.append(y_seq)
        
        # Convert to tensors
        x = torch.from_numpy(np.array(x_data).astype(np.int64)).to(self.device)
        y = torch.from_numpy(np.array(y_data).astype(np.int64)).to(self.device)
        
        return x, y


def get_lr(iter_num, config):
    """Learning rate schedule for SFT."""
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters
    
    plateau_iters = int(config.lr_decay_iters * 0.5)  # Shorter plateau for SFT
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
    """Evaluate model with SFT loss (only on non-masked tokens)."""
    model.eval()
    losses = []
    
    for _ in range(config.eval_iters):
        x, y = data_loader.get_batch('val', config.batch_size)
        
        if config.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                logits, _ = model(x)  # Don't use model's loss calculation
        else:
            logits, _ = model(x)
        
        # Compute SFT loss (only on non-masked tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100  # Ignore masked tokens
        )
        
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def create_optimizer_groups(model, weight_decay):
    """
    Create parameter groups with proper weight decay exclusions.
    
    Excludes from weight decay:
    - LayerNorms (RMSNorm in our case)
    - Biases (if any)
    - Embeddings and positional embeddings
    
    This prevents regularization from hurting these sensitive parameters.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter should be excluded from weight decay
        if any(nd in name for nd in ['ln_', 'norm', 'bias', 'wte', 'wpe', 'rope_cache']):
            # LayerNorms, biases, embeddings - no weight decay
            no_decay_params.append(param)
        else:
            # Linear layers, attention weights - apply weight decay
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Report statistics
    num_decay = sum(p.numel() for p in decay_params)
    num_no_decay = sum(p.numel() for p in no_decay_params)
    total_params = num_decay + num_no_decay
    
    print(f"Optimizer groups created:")
    print(f"  With weight decay: {num_decay:,} params ({100*num_decay/total_params:.1f}%)")
    print(f"  No weight decay: {num_no_decay:,} params ({100*num_no_decay/total_params:.1f}%)")
    
    return param_groups

def load_pretrained_model(checkpoint_path, config, device='cuda'):
    """Load the pretrained elite model."""
    print(f"Loading pretrained model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model config from checkpoint
    pretrained_config = checkpoint['config']
    model_config = ModelConfig(
        vocab_size=pretrained_config['vocab_size'],
        n_layer=pretrained_config['n_layer'],
        n_head=pretrained_config['n_head'],
        n_embd=pretrained_config['n_embd'],
        n_kv_heads=pretrained_config['n_kv_heads'],
        block_size=pretrained_config['block_size'],
        dropout=config.dropout,  # Use SFT dropout
        use_fp8=config.use_fp8,
        fp8_amax_history_len=config.fp8_amax_history_len,
    )
    
    # Create model and load weights
    model = CleanGPT_TE(model_config).to(device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(torch.bfloat16)
    
    print(f"✓ Loaded elite model!")
    print(f"  Training iterations completed: {checkpoint['iter_num']}")
    print(f"  Final validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Parameters: {model.num_parameters()/1e6:.1f}M")
    
    return model, model_config


def save_checkpoint(model, optimizer, config, iter_num, val_loss, checkpoint_path):
    """Save SFT checkpoint."""
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': asdict(config),
        'iter_num': iter_num,
        'val_loss': val_loss,
        'training_type': 'sft',  # Mark as SFT checkpoint
    }
    
    print(f"Saving SFT checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)


def train():
    """Main SFT training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--data_dir', type=str, default='data_sft', help='SFT data directory')
    parser.add_argument('--base_model', type=str, default='best_model_fp8_optimized.pt', help='Base pretrained model')
    parser.add_argument('--max_iters', type=int, default=800, help='Max iterations (default: 800 = 1 epoch)')
    parser.add_argument('--learning_rate', type=float, default=7e-5, help='Learning rate (default: 7e-5, balanced)')
    parser.add_argument('--eval_interval', type=int, default=50, help='Eval interval (default: 50)')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--no_fp8', action='store_true', help='Disable FP8')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    args = parser.parse_args()
    
    # Configuration
    config = SFTConfig()
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.grad_accum:
        config.gradient_accumulation_steps = args.grad_accum
    if args.no_fp8:
        config.use_fp8 = False
    
    config.data_dir = args.data_dir
    config.base_model = args.base_model
    config.max_iters = args.max_iters
    config.learning_rate = args.learning_rate
    config.eval_interval = args.eval_interval
    config.log_interval = args.log_interval
    
    # Load pretrained model
    model, model_config = load_pretrained_model(config.base_model, config, config.device)
    
    # Get FP8 recipe
    fp8_recipe = get_fp8_recipe(model_config)
    
    # Create parameter groups with proper weight decay exclusions
    param_groups = create_optimizer_groups(model, config.weight_decay)
    
    # Optimizer - fresh start for SFT with parameter groups
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=True  # Use fused optimizer for speed
    )
    
    # Data loader
    data_loader = SFTDataLoader(config.data_dir, config.block_size, config.device)
    
    # Initialize WandB logger
    logger = WandBLogger(
        enabled=not args.no_wandb,
        project=config.wandb_project,
        run_name=config.wandb_run_name or f"sft_elite_{config.learning_rate:.0e}_{config.batch_size}b",
        config=asdict(config)
    )
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    # Training info
    print("\n" + "="*60)
    print("SFT TRAINING - ELITE MODEL CONTINUATION")
    print("="*60)
    print(f"Base model: {config.base_model}")
    print(f"Architecture: {model_config.n_layer}L, {model_config.n_head}H, {model_config.n_embd}D")
    print(f"Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"SFT Config:")
    print(f"  - Learning rate: {config.learning_rate:.2e} (balanced)")
    print(f"  - Weight decay: {config.weight_decay} (excludes norms/bias)")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  - Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  - Max iterations: {config.max_iters}")
    print(f"  - FP8: {config.use_fp8}")
    print(f"Data:")
    print(f"  - Train examples: {data_loader.metadata['train_examples']}")
    print(f"  - Train tokens: {data_loader.metadata['train_tokens']:,}")
    print("="*60 + "\n")
    
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
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Accumulate gradients
        total_loss = 0
        
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = data_loader.get_batch('train', config.batch_size)
            
            # Determine if using FP8
            use_fp8_now = config.use_fp8 and (iter_num >= config.fp8_warmup_steps)
            
            # Forward pass
            if use_fp8_now:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, _ = model(x)  # Don't use model's loss
            else:
                logits, _ = model(x)
            
            # Compute SFT loss (only on assistant responses)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100  # Critical: ignore masked tokens
            )
            
            total_loss += loss.item()
            loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        
        # Optimizer step
        optimizer.step()
        
        # Update token count (only count non-masked tokens)
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
            
            # Log metrics
            logger.log_metrics({
                'sft/loss': avg_loss,
                'sft/perplexity': perplexity,
                'sft/lr': lr,
                'sft/tokens_per_sec': tokens_per_sec,
                'sft/grad_norm': grad_norm.item() if config.grad_clip > 0 else 0,
                'sft/fp8_active': use_fp8_now,
                'sft/iteration': iter_num,
            }, step=iter_num)
            
            tokens_processed = 0
            t0 = time.time()
        
        # Evaluation
        if iter_num % config.eval_interval == 0:
            val_loss = evaluate(model, data_loader, config, fp8_recipe)
            print(f"Step {iter_num}: val loss {val_loss:.4f}")
            
            # Calculate validation perplexity
            try:
                val_perplexity = math.exp(min(20.0, val_loss))
            except:
                val_perplexity = float('inf')
            
            # Log validation metrics
            logger.log_metrics({
                'val/loss': val_loss,
                'val/perplexity': val_perplexity,
            }, step=iter_num)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.set_summary(
                    best_val_loss=val_loss,
                    best_val_perplexity=val_perplexity,
                    best_iter=iter_num
                )
                save_checkpoint(
                    model, optimizer, config, iter_num, val_loss,
                    Path(config.checkpoint_dir) / 'best_sft_model.pt'
                )
        
        # Regular checkpoints
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            save_checkpoint(
                model, optimizer, config, iter_num, val_loss,
                Path(config.checkpoint_dir) / f'sft_checkpoint_{iter_num}.pt'
            )
    
    # Final evaluation
    final_val_loss = evaluate(model, data_loader, config, fp8_recipe)
    try:
        final_perplexity = math.exp(min(20.0, final_val_loss))
    except:
        final_perplexity = float('inf')
    
    logger.set_summary(
        final_val_loss=final_val_loss,
        final_perplexity=final_perplexity,
        total_iterations=config.max_iters
    )
    
    print("\n" + "="*60)
    print("SFT TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final perplexity: {final_perplexity:.2f}")
    print(f"Elite model successfully fine-tuned!")
    print("="*60)
    
    # Clean up wandb
    logger.finish()


if __name__ == "__main__":
    train()