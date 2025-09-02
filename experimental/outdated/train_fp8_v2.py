#!/usr/bin/env python3
"""
FP8 training script (v2) that uses the cuDNN DPA (FP8) attention path
implemented in model_te_v2.CleanGPT_TE.

Key differences from the clean version:
- Imports the v2 model which uses TE cuDNN DotProductAttention for FP8 attention
  and native GQA (no KV duplication).
- Keeps te.fp8_autocast gating and the simple training loop.
"""

import os
import math
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import argparse

import torch
import torch.nn.functional as F
import numpy as np

# Force TE to prefer cuDNN/DPA attention backend (0 = cuDNN, 1 = Flash)
# Do this BEFORE importing any TE modules or the model.
os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"

# Import v2 model (FP8 attention via TE cuDNN DPA)
from model_te_v2 import ModelConfig, CleanGPT_TE, get_fp8_recipe

# TransformerEngine
import transformer_engine.pytorch as te
try:
    import transformer_engine as te_base  # for version
    _te_version = getattr(te_base, "__version__", "unknown")
except Exception:
    te_base = None
    _te_version = "unknown"

# WandB logger
from wandb_logger import WandBLogger


@dataclass
class TrainingConfig:
    # Model config
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 49152
    n_kv_heads: int = 3
    block_size: int = 2048
    dropout: float = 0.05

    # FP8 config
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_warmup_steps: int = 100

    # Simplified knobs
    fuse_wgrad_accumulation: bool = False
    cache_fp8_weights: bool = False

    # Training config
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    lr_decay_iters: int = 2000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999

    # System
    device: str = "cuda"
    compile: bool = False
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints_fp8_v2"
    data_dir: str = "data_mixed_3b"

    # Wandb
    wandb_project: str = "fp8-v2"
    wandb_run_name: Optional[str] = None


class DataLoader:
    """Memory-mapped data loader."""

    def __init__(self, data_dir, block_size, device="cuda"):
        self.block_size = block_size
        self.device = device

        data_path = Path(data_dir)
        self.train_data = np.memmap(data_path / "train.bin", dtype=np.uint16, mode="r")
        self.val_data = np.memmap(data_path / "val.bin", dtype=np.uint16, mode="r")

        # Precompute offsets once to save per-step allocations
        self._offsets = torch.arange(self.block_size + 1)

        print(f"Loaded data from {data_dir}")
        print(f"  Train: {len(self.train_data):,} tokens")
        print(f"  Val: {len(self.val_data):,} tokens")

    def get_batch(self, split, batch_size):
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise RuntimeError(
                f"Dataset too small for block_size={self.block_size}; len={len(data)}"
            )
        ix = torch.randint(max_start, (batch_size,))

        indices = ix.unsqueeze(1) + self._offsets.unsqueeze(0)
        # Keep as uint16 on CPU, then transfer and cast on device
        batch_np = np.asarray(data[indices.numpy()], dtype=np.uint16)
        batch = torch.from_numpy(batch_np).pin_memory()

        x = batch[:, :-1].to(self.device, non_blocking=True).to(torch.long)
        y = batch[:, 1:].to(self.device, non_blocking=True).to(torch.long)
        return x, y

    def get_batch_cpu_uint16(self, split, batch_size):
        """Return pinned CPU uint16 tensors (x16, y16). Device transfer is caller's responsibility."""
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise RuntimeError(
                f"Dataset too small for block_size={self.block_size}; len={len(data)}"
            )
        ix = torch.randint(max_start, (batch_size,))
        indices = ix.unsqueeze(1) + self._offsets.unsqueeze(0)
        batch_np = np.asarray(data[indices.numpy()], dtype=np.uint16)
        batch = torch.from_numpy(batch_np).pin_memory()
        return batch[:, :-1], batch[:, 1:]


class Prefetcher:
    """Asynchronous GPU prefetcher to overlap H2D copies with compute."""

    def __init__(self, data_loader: DataLoader, batch_size: int, device: str = "cuda"):
        self.dl = data_loader
        self.bs = batch_size
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._next = None
        self._event = None

    def preload(self):
        if self.stream is None:
            # CPU only fallback
            x16, y16 = self.dl.get_batch_cpu_uint16("train", self.bs)
            x = x16.to(self.device).to(torch.long)
            y = y16.to(self.device).to(torch.long)
            self._next = (x, y)
            self._event = None
            return

        x16, y16 = self.dl.get_batch_cpu_uint16("train", self.bs)
        with torch.cuda.stream(self.stream):
            x = x16.to(self.device, non_blocking=True).to(torch.long)
            y = y16.to(self.device, non_blocking=True).to(torch.long)
        evt = torch.cuda.Event()
        evt.record(self.stream)
        self._next = (x, y)
        self._event = evt

    def next(self):
        if self._next is None:
            self.preload()
        # Synchronize current stream with prefetch stream if needed
        if self._event is not None:
            torch.cuda.current_stream().wait_event(self._event)
        x, y = self._next
        # Schedule next prefetch immediately
        self.preload()
        return x, y


def get_lr(iter_num, config: TrainingConfig):
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
def evaluate(model, data_loader, config: TrainingConfig, fp8_recipe):
    model.eval()
    losses = []

    for _ in range(config.eval_iters):
        x, y = data_loader.get_batch("val", config.batch_size)
        if config.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return np.mean(losses)


def save_checkpoint(model, optimizer, config: TrainingConfig, iter_num, val_loss, checkpoint_path):
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "iter_num": iter_num,
        "val_loss": val_loss,
    }
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="data_mixed_3b", help="Data directory")
    parser.add_argument("--max_iters", type=int, default=2000, help="Max iterations")
    parser.add_argument("--eval_interval", type=int, default=200, help="Eval interval")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--no_fp8", action="store_true", help="Disable FP8")
    parser.add_argument(
        "--attention",
        type=str,
        default="auto",
        choices=["auto", "dpa", "sdpa"],
        help="Attention backend selection: auto tries TE DPA then falls back; dpa forces TE DPA; sdpa forces PyTorch SDPA",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    # Configuration
    config = TrainingConfig()
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_fp8:
        config.use_fp8 = False
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
        fuse_wgrad_accumulation=False,
    )

    # Create v2 model
    model = CleanGPT_TE(model_config).to(config.device)
    model = model.to(torch.bfloat16)

    # Prefer FlashAttention path for PyTorch SDPA when used
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    except Exception:
        pass
    # Enable TF32 for additional speed on Ada/Ampere
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # FP8 recipe
    fp8_recipe = get_fp8_recipe(model_config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=True,
    )

    # Data loader
    data_loader = DataLoader(config.data_dir, config.block_size, config.device)

    # WandB logger
    logger = WandBLogger(
        enabled=not args.no_wandb,
        project=config.wandb_project,
        run_name=config.wandb_run_name or f"fp8_v2_{config.n_layer}L_{config.batch_size}b",
        config=asdict(config),
    )
    logger.watch(model, log="gradients", log_freq=100)

    # Checkpoint directory
    Path(config.checkpoint_dir).mkdir(exist_ok=True)

    # Training info
    print("\n" + "=" * 50)
    print("Starting FP8 Training (v2: cuDNN DPA attention)")
    print("=" * 50)
    print(f"Model: {model_config.n_layer}L, {model_config.n_head}H, {model_config.n_embd}D")
    print(f"Parameters: {model.num_parameters()/1e6:.1f}M")
    try:
        cudnn_ver = torch.backends.cudnn.version()
    except Exception:
        cudnn_ver = None
    print(f"Environment: TE={_te_version}, CUDA={torch.version.cuda}, cuDNN={cudnn_ver}")
    print("Status:")
    print(f"  - FP8 enabled: {config.use_fp8}")
    # Respect attention selection
    if args.attention == "sdpa":
        model.set_attention_backend("sdpa")
    elif args.attention == "dpa":
        model.set_attention_backend("te_dpa")
    # Report
    _backend_now = model.get_attention_backend()
    attn_backend = "TE cuDNN DPA (native GQA)" if _backend_now == "te_dpa" else "PyTorch SDPA (BF16 fallback)"
    print(f"  - Attention: {attn_backend} (requested: {args.attention})")
    print("  - Expected: higher tok/s than SDPA BF16")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print("=" * 50 + "\n")

    model.train()
    iter_num = 0
    best_val_loss = float("inf")
    t0 = time.time()
    tokens_processed = 0

    # Auto-switch state
    baseline_tps_samples = []  # tokens/sec when FP8 is off
    fp8_tps_samples = []       # tokens/sec when FP8 is on
    switched_to_sdpa = False

    for iter_num in range(config.max_iters):
        # LR schedule
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        # Prefetcher: overlap copies with compute for more stable throughput
        prefetcher = Prefetcher(data_loader, config.batch_size, config.device)
        x, y = prefetcher.next()
        for micro_step in range(config.gradient_accumulation_steps):
            if micro_step > 0:
                x, y = prefetcher.next()

            use_fp8_now = config.use_fp8 and (iter_num >= config.fp8_warmup_steps)
            if use_fp8_now:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)

            total_loss += loss.item()
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

        # Clip gradients
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()

        # Tokens processed in this interval
        tokens_processed += (
            config.batch_size * config.block_size * config.gradient_accumulation_steps
        )

        # Logging
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = tokens_processed / dt if dt > 0 else 0.0
            avg_loss = total_loss / config.gradient_accumulation_steps
            print(
                f"iter {iter_num}: loss {avg_loss:.4f}, lr {lr:.2e}, "
                f"{tokens_per_sec/1e3:.1f}k tok/s, FP8: {use_fp8_now}"
            )
            try:
                perplexity = math.exp(min(20.0, avg_loss))
            except Exception:
                perplexity = float("inf")

            # Track baseline vs fp8 throughput for auto mode
            if args.attention == "auto":
                if use_fp8_now:
                    fp8_tps_samples.append(tokens_per_sec)
                else:
                    baseline_tps_samples.append(tokens_per_sec)

                # After enough samples, if FP8+DPA underperforms baseline by >10%, switch to SDPA
                if (
                    not switched_to_sdpa
                    and len(baseline_tps_samples) >= 3
                    and len(fp8_tps_samples) >= 3
                    and model.get_attention_backend() == "te_dpa"
                ):
                    base_avg = sum(baseline_tps_samples[-3:]) / 3
                    fp8_avg = sum(fp8_tps_samples[-3:]) / 3
                    if fp8_avg < 0.9 * base_avg:
                        print(
                            f"[train_fp8_v2] Auto-switch: FP8 DPA avg {fp8_avg/1e3:.1f}k < 0.9 * baseline {base_avg/1e3:.1f}k; switching to SDPA"
                        )
                        model.set_attention_backend("sdpa")
                        switched_to_sdpa = True

            logger.log_metrics(
                {
                    "train/loss": avg_loss,
                    "train/perplexity": perplexity,
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/grad_norm": grad_norm.item() if config.grad_clip > 0 else 0,
                    "train/fp8_active": use_fp8_now,
                    "train/gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available()
                    else 0,
                    "train/iteration": iter_num,
                },
                step=iter_num,
            )
            tokens_processed = 0
            t0 = time.time()

        # Evaluation
        if iter_num % config.eval_interval == 0:
            val_loss = evaluate(model, data_loader, config, fp8_recipe)
            print(f"Step {iter_num}: val loss {val_loss:.4f}")
            try:
                val_perplexity = math.exp(min(20.0, val_loss))
            except Exception:
                val_perplexity = float("inf")

            logger.log_metrics(
                {"val/loss": val_loss, "val/perplexity": val_perplexity},
                step=iter_num,
            )

            if val_loss < best_val_loss:
                logger.set_summary(
                    best_val_loss=val_loss,
                    best_val_perplexity=val_perplexity,
                    best_iter=iter_num,
                )
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    config,
                    iter_num,
                    val_loss,
                    Path(config.checkpoint_dir) / "best_model_fp8_v2.pt",
                )

            # Do not count eval time in throughput
            tokens_processed = 0
            t0 = time.time()

        # Regular checkpoints
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            save_checkpoint(
                model,
                optimizer,
                config,
                iter_num,
                best_val_loss,
                Path(config.checkpoint_dir) / f"checkpoint_{iter_num}_fp8_v2.pt",
            )

    # Final summary
    final_val_loss = evaluate(model, data_loader, config, fp8_recipe)
    try:
        final_perplexity = math.exp(min(20.0, final_val_loss))
    except Exception:
        final_perplexity = float("inf")

    logger.set_summary(
        final_val_loss=final_val_loss,
        final_perplexity=final_perplexity,
        total_iterations=config.max_iters,
        model_params=model.num_parameters(),
    )

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final perplexity: {final_perplexity:.2f}")
    print("v2: TE cuDNN DPA attention (FP8)")
    print("=" * 50)

    logger.finish()


if __name__ == "__main__":
    train()
