"""
V2 MoE training script with all optimizations applied.
Key improvements:
- No MTP overhead
- Optimized batch sizes
- Comprehensive timing
- Better monitoring
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Optional, Dict
import time
import argparse
from contextlib import nullcontext

from model_moe_v2 import MoEConfig, MoEModelV2


class DataLoader:
    """Simple data loader for pretraining."""
    
    def __init__(self, data_dir: str, batch_size: int, seq_len: int, device: str = "cuda"):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        
        # Load tokenized data
        self.train_tokens = np.memmap(
            os.path.join(data_dir, "train_tokens.bin"),
            dtype=np.uint16,
            mode='r'
        )
        self.val_tokens = np.memmap(
            os.path.join(data_dir, "val_tokens.bin"),
            dtype=np.uint16,
            mode='r'
        )
        
        print(f"Loaded training data: {len(self.train_tokens) / 1e6:.1f}M tokens")
        print(f"Loaded validation data: {len(self.val_tokens) / 1e6:.1f}M tokens")
    
    def get_batch(self, split: str):
        """Get a batch of data."""
        data = self.train_tokens if split == "train" else self.val_tokens
        
        # Random positions
        ix = torch.randint(len(data) - self.seq_len, (self.batch_size,))
        
        # Gather sequences
        x = torch.stack([
            torch.from_numpy(data[i:i+self.seq_len].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(data[i+1:i+1+self.seq_len].astype(np.int64))
            for i in ix
        ])
        
        return x.to(self.device), y.to(self.device)


class Trainer:
    """V2 MoE trainer with optimized settings."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Model configuration
        model_config = MoEConfig(
            n_layer=config["n_layer"],
            n_embd=config["n_embd"],
            n_head=config["n_head"],
            n_kv_head=config["n_kv_head"],
            num_experts=config["num_experts"],
            expert_expansion=config["expert_expansion"],
            capacity_factor=config["capacity_factor"],
            overflow_policy=config["overflow_policy"],
            router_aux_loss_weight=config["router_aux_loss_weight"],
            dropout=config["dropout"],
            log_interval=config["log_interval"],
        )
        
        # Initialize model
        self.model = MoEModelV2(model_config).cuda()
        
        # Data loader
        self.data_loader = DataLoader(
            config["data_dir"],
            config["batch_size"],
            model_config.block_size
        )
        
        # Optimizer
        self.optimizer = self.configure_optimizer()
        
        # Mixed precision
        self.use_amp = config["use_amp"]
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum_steps = config["grad_accum_steps"]
        
        # Logging
        self.step = 0
        self.tokens_seen = 0
        self.log_interval = config["log_interval"]
        self.eval_interval = config["eval_interval"]
        
        # Timing
        self.timings = {
            "forward": [],
            "backward": [],
            "optimizer": [],
            "total": [],
        }
        
        # Wandb
        self.use_wandb = config.get("wandb", False)
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config.get("wandb_project", "moe-v2"),
                name=config.get("wandb_run_name", None),
                config=config
            )
    
    def configure_optimizer(self):
        """Configure AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ["bias", "ln", "layernorm"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        optimizer = AdamW(
            optim_groups,
            lr=self.config["learning_rate"],
            betas=(self.config["beta1"], self.config["beta2"]),
            eps=self.config["adam_eps"],
        )
        
        print(f"Optimizer configured:")
        print(f"  Decay params: {sum(p.numel() for p in decay_params) / 1e6:.1f}M")
        print(f"  No-decay params: {sum(p.numel() for p in no_decay_params) / 1e6:.1f}M")
        
        return optimizer
    
    def get_lr(self) -> float:
        """Get current learning rate with cosine schedule."""
        # Warmup
        if self.step < self.config["warmup_steps"]:
            return self.config["learning_rate"] * (self.step + 1) / self.config["warmup_steps"]
        
        # Cosine decay
        if self.step > self.config["max_steps"]:
            return self.config["min_lr"]
        
        decay_ratio = (self.step - self.config["warmup_steps"]) / (
            self.config["max_steps"] - self.config["warmup_steps"]
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        
        return self.config["min_lr"] + coeff * (
            self.config["learning_rate"] - self.config["min_lr"]
        )
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()
        losses = []
        
        for _ in range(self.config["eval_iters"]):
            x, y = self.data_loader.get_batch("val")
            
            with autocast() if self.use_amp else nullcontext():
                _, loss = self.model(x, y)
            
            losses.append(loss.item())
        
        self.model.train()
        
        mean_loss = np.mean(losses)
        return {
            "val_loss": mean_loss,
            "val_perplexity": np.exp(mean_loss),
        }
    
    def train_step(self):
        """Single training step with timing."""
        step_start = time.time()
        
        # Update learning rate
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Accumulate gradients
        total_loss = 0.0
        forward_time = 0.0
        backward_time = 0.0
        
        for micro_step in range(self.grad_accum_steps):
            # Get batch
            x, y = self.data_loader.get_batch("train")
            
            # Forward pass with timing
            forward_start = time.time()
            with autocast() if self.use_amp else nullcontext():
                _, loss = self.model(x, y)
                loss = loss / self.grad_accum_steps
            forward_time += time.time() - forward_start
            
            total_loss += loss.item()
            
            # Backward pass with timing
            backward_start = time.time()
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            backward_time += time.time() - backward_start
        
        # Optimizer step with timing
        optimizer_start = time.time()
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            self.optimizer.step()
        optimizer_time = time.time() - optimizer_start
        
        # Total time
        total_time = time.time() - step_start
        
        # Update timings
        self.timings["forward"].append(forward_time * 1000)
        self.timings["backward"].append(backward_time * 1000)
        self.timings["optimizer"].append(optimizer_time * 1000)
        self.timings["total"].append(total_time * 1000)
        
        # Update counters
        self.step += 1
        self.tokens_seen += self.config["batch_size"] * self.model.config.block_size * self.grad_accum_steps
        
        # Log metrics
        if self.step % self.log_interval == 0:
            self.log_metrics(total_loss, lr)
        
        # Evaluate
        if self.step % self.eval_interval == 0:
            val_metrics = self.evaluate()
            print(f"\n[Step {self.step}] Validation:")
            print(f"  Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Perplexity: {val_metrics['val_perplexity']:.2f}")
            
            if self.use_wandb:
                import wandb
                wandb.log(val_metrics, step=self.step)
    
    def log_metrics(self, loss: float, lr: float):
        """Log training metrics."""
        # Compute averages
        avg_forward = np.mean(self.timings["forward"][-self.log_interval:])
        avg_backward = np.mean(self.timings["backward"][-self.log_interval:])
        avg_optimizer = np.mean(self.timings["optimizer"][-self.log_interval:])
        avg_total = np.mean(self.timings["total"][-self.log_interval:])
        
        # Tokens per second
        tokens_per_sec = (
            self.config["batch_size"] * 
            self.model.config.block_size * 
            self.grad_accum_steps * 
            1000 / avg_total
        )
        
        print(f"\n[Step {self.step}] Training Metrics:")
        print(f"  Loss: {loss:.4f} | Perplexity: {np.exp(loss):.2f}")
        print(f"  LR: {lr:.6f}")
        print(f"  Tokens: {self.tokens_seen/1e6:.1f}M")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
        print(f"  Timing (ms): fwd={avg_forward:.1f}, bwd={avg_backward:.1f}, "
              f"opt={avg_optimizer:.1f}, total={avg_total:.1f}")
        
        if self.use_wandb:
            import wandb
            wandb.log({
                "train/loss": loss,
                "train/perplexity": np.exp(loss),
                "train/lr": lr,
                "train/tokens": self.tokens_seen,
                "train/tokens_per_sec": tokens_per_sec,
                "timing/forward_ms": avg_forward,
                "timing/backward_ms": avg_backward,
                "timing/optimizer_ms": avg_optimizer,
                "timing/total_ms": avg_total,
            }, step=self.step)
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training:")
        print(f"  Total steps: {self.config['max_steps']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Gradient accumulation: {self.grad_accum_steps}")
        print(f"  Effective batch size: {self.config['batch_size'] * self.grad_accum_steps}")
        
        self.model.train()
        
        while self.step < self.config["max_steps"]:
            self.train_step()
            
            # Save checkpoint
            if self.step % self.config["checkpoint_interval"] == 0:
                self.save_checkpoint()
        
        print(f"\nTraining completed!")
        print(f"  Final step: {self.step}")
        print(f"  Total tokens: {self.tokens_seen/1e9:.2f}B")
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "config": self.config,
        }
        
        if self.use_amp:
            checkpoint["scaler_state"] = self.scaler.state_dict()
        
        checkpoint_path = f"checkpoints/moe_v2_step_{self.step}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train V2 MoE model")
    
    # Model configuration
    parser.add_argument("--n_layer", type=int, default=24, help="Number of layers")
    parser.add_argument("--n_embd", type=int, default=896, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=28, help="Number of attention heads")
    parser.add_argument("--n_kv_head", type=int, default=7, help="Number of KV heads (GQA)")
    
    # MoE configuration
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")
    parser.add_argument("--expert_expansion", type=float, default=3.5, help="Expert FFN expansion")
    parser.add_argument("--capacity_factor", type=float, default=1.0, help="Switch capacity factor")
    parser.add_argument("--overflow_policy", type=str, default="drop", choices=["drop", "rescue"])
    parser.add_argument("--router_aux_loss_weight", type=float, default=0.01, help="Auxiliary loss weight")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Schedule
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    
    # Optimization
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--eval_iters", type=int, default=20, help="Evaluation iterations")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="moe-v2", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    
    args = parser.parse_args()
    
    # Convert to config dict
    config = vars(args)
    
    # Print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()