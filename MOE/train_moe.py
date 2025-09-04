"""
FP8 training script for the GLM-mini-ish PR-MoE model.
- Uses Transformer Engine FP8 autocast
- Works with memmapped token datasets (train.bin/val.bin of uint16 ids)
- Logs MoE-specific stats (per-expert EMA load, avg active experts per token)
- Simple, fast, and close to your previous dense script's structure
"""

import os
import math
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure we can import the provided model file
import sys
THIS_DIR = Path(__file__).parent.resolve()
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

# Import the MoE model
from model_moe import (
    ModelConfig,
    OptimizedGPT_GLMini_PRMoE,
    get_fp8_recipe,
    build_glm_mini_prmoe_mtp,
    PyramidResidualMoE,
)

# Transformer Engine
import transformer_engine.pytorch as te

# WandB logger (MoE-capable)
from wandb_logger_moe import WandBLogger


@dataclass
class TrainingConfig:
    # Data & batching
    data_dir: str = "data_mixed_3b"     # folder with train.bin / val.bin (uint16)
    block_size: int = 2048              # Stage 1 pretraining: shorter sequences
    batch_size: int = 8
    grad_accum_steps: int = 16

    # Optim & schedule
    learning_rate: float = 6e-4
    min_lr: float = 1e-4
    warmup_iters: int = 2000
    max_iters: int = 20000
    lr_decay_iters: int = 80000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0

    # Eval & logging & ckpts
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 50
    checkpoint_interval: int = 2000
    checkpoint_dir: str = "checkpoints_glm_prmoe"

    # System
    device: str = "cuda"
    compile: bool = False  # keep False with TE
    seed: int = 1337

    # FP8
    use_fp8: bool = True
    fp8_amax_history_len: int = 16
    fp8_warmup_steps: int = 1000
    fp8_amax_compute_algo: str = "max"

    # Wandb
    wandb_project: str = "glm-mini-prmoe"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None  # "offline" to force offline


class DataLoader:
    """Memory-mapped loader for token id arrays (uint16)."""
    def __init__(self, data_dir: str, block_size: int, device: str = "cuda"):
        self.block_size = block_size
        self.device = device
        data_path = Path(data_dir)
        self.train_data = np.memmap(data_path / "train.bin", dtype=np.uint16, mode="r")
        self.val_data = np.memmap(data_path / "val.bin", dtype=np.uint16, mode="r")
        print(f"Loaded data from {data_dir}")
        print(f"  Train: {len(self.train_data):,} tokens")
        print(f"  Val:   {len(self.val_data):,} tokens")

    def get_batch(self, split: str, batch_size: int):
        data = self.train_data if split == "train" else self.val_data
        L = len(data) - self.block_size - 1
        ix = torch.randint(L, (batch_size,))
        # vectorized gather
        offsets = torch.arange(self.block_size + 1)
        idx = ix.unsqueeze(1) + offsets.unsqueeze(0)
        batch = torch.from_numpy(data[idx.numpy()].astype(np.int64)).pin_memory()
        x = batch[:, :-1].to(self.device, non_blocking=True)
        y = batch[:, 1:].to(self.device, non_blocking=True)
        return x, y


def cosine_decay(iter_num: int, cfg: TrainingConfig):
    if iter_num < cfg.warmup_iters:
        return cfg.learning_rate * (iter_num / max(1, cfg.warmup_iters))
    # cosine to min_lr over lr_decay_iters
    t = min(1.0, (iter_num - cfg.warmup_iters) / max(1, (cfg.lr_decay_iters - cfg.warmup_iters)))
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def evaluate(model, data_loader: DataLoader, tcfg: TrainingConfig, fp8_recipe):
    model.eval()
    losses = []
    for _ in range(tcfg.eval_iters):
        x, y = data_loader.get_batch("val", tcfg.batch_size)
        if tcfg.use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# --------- MoE monitoring (monkeypatch router to record batch stats) ---------
def attach_router_monitor(model: nn.Module):
    """
    Wrap PyramidResidualMoE._route_sigmoid to record:
      - last_k_mean: avg active experts per token (scalar)
      - last_loads: per-expert fraction this batch (tensor [E])
    """
    for m in model.modules():
        if isinstance(m, PyramidResidualMoE):
            if hasattr(m, "_route_sigmoid_raw"):
                continue
            m._route_sigmoid_raw = m._route_sigmoid

            def _route_sigmoid_monitored(self, x):
                weights, active = self._route_sigmoid_raw(x)
                with torch.no_grad():
                    # avg k across tokens
                    k_mean = active.sum(dim=-1).float().mean().item()
                    self.last_k_mean = k_mean
                    # per-expert fraction
                    B, T, E = active.shape[0], active.shape[1], active.shape[2]
                    loads = active.sum(dim=(0,1)).float() / (B*T)
                    self.last_loads = loads.detach().cpu()
                return weights, active

            # bind the method
            m._route_sigmoid = _route_sigmoid_monitored.__get__(m, m.__class__)


def collect_moe_stats(model: nn.Module) -> Dict[str, float]:
    """Aggregate stats across all MoE layers if present."""
    k_means: List[float] = []
    load_stds: List[float] = []
    for m in model.modules():
        if isinstance(m, PyramidResidualMoE):
            if hasattr(m, "last_k_mean"):
                k_means.append(float(m.last_k_mean))
            if hasattr(m, "last_loads"):
                ls = torch.as_tensor(m.last_loads)
                if ls.numel() > 0:
                    load_stds.append(float(ls.std().item()))
    out = {}
    if k_means:
        out["moe/k_mean"] = float(np.mean(k_means))
    if load_stds:
        out["moe/load_std_mean"] = float(np.mean(load_stds))
    return out


def save_checkpoint(model, optimizer, tcfg: TrainingConfig, iter_num: int, val_loss: float, path: Path):
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_config": asdict(tcfg),
        "iter_num": iter_num,
        "val_loss": val_loss,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_mixed_3b")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--no_fp8", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    tcfg = TrainingConfig()
    if args.batch_size: tcfg.batch_size = args.batch_size
    if args.max_iters:  tcfg.max_iters  = args.max_iters
    if args.eval_interval: tcfg.eval_interval = args.eval_interval
    if args.no_fp8: tcfg.use_fp8 = False
    if args.run_name: tcfg.wandb_run_name = args.run_name
    tcfg.data_dir = args.data_dir

    torch.manual_seed(tcfg.seed)
    torch.cuda.manual_seed_all(tcfg.seed)

    # Build model (GLM-mini-ish defaults)
    model, mcfg = build_glm_mini_prmoe_mtp()
    # Ensure block_size matches the training config
    mcfg.block_size = tcfg.block_size
    model = model.to(tcfg.device).to(torch.bfloat16)

    # Attach MoE router monitor (records per-batch stats)
    attach_router_monitor(model)

    # FP8 recipe
    fp8_recipe = get_fp8_recipe(amax_history_len=tcfg.fp8_amax_history_len, amax_compute_algo=tcfg.fp8_amax_compute_algo)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.learning_rate,
        betas=(tcfg.beta1, tcfg.beta2),
        weight_decay=tcfg.weight_decay,
        fused=True
    )

    # Data
    loader = DataLoader(tcfg.data_dir, tcfg.block_size, tcfg.device)

    # Logger
    logger = WandBLogger(
        enabled=not args.no_wandb,
        project=tcfg.wandb_project,
        run_name=tcfg.wandb_run_name or f"glm_prmoe_{mcfg.n_layer}L_{tcfg.batch_size}b",
        entity=tcfg.wandb_entity,
        mode=tcfg.wandb_mode,
        config={**asdict(tcfg), "model": asdict(mcfg)}
    )

    # Training loop
    iter_num = 0
    best_val = float("inf")
    tokens_processed = 0
    t0 = time.time()

    model.train()
    for iter_num in range(tcfg.max_iters):
        # LR schedule
        lr = cosine_decay(iter_num, tcfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro in range(tcfg.grad_accum_steps):
            x, y = loader.get_batch("train", tcfg.batch_size)

            use_fp8_now = tcfg.use_fp8 and (iter_num >= tcfg.fp8_warmup_steps)
            if use_fp8_now:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)

            total_loss += loss.item()
            (loss / tcfg.grad_accum_steps).backward()

        # clip
        grad_norm = 0.0
        if tcfg.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip))

        optimizer.step()

        tokens_processed += tcfg.batch_size * tcfg.block_size * tcfg.grad_accum_steps

        # Logging
        if iter_num % tcfg.log_interval == 0:
            dt = max(1e-6, (time.time() - t0))
            tps = tokens_processed / dt
            avg_loss = total_loss / tcfg.grad_accum_steps
            # For MTP, normalize by total weight (2.75) to get correct perplexity
            mtp_weight = 2.75 if mcfg.use_mtp else 1.0
            normalized_loss = avg_loss / mtp_weight
            ppl = math.exp(min(20.0, normalized_loss))

            # collect MoE stats
            moe_stats = collect_moe_stats(model)

            msg = f"it {iter_num}: loss {avg_loss:.4f} | ppl {ppl:.2f} | lr {lr:.2e} | {tps/1e3:.1f}k tok/s | fp8 {use_fp8_now}"
            if "moe/k_mean" in moe_stats:
                msg += f" | k̄ {moe_stats['moe/k_mean']:.2f}"
            if "moe/load_std_mean" in moe_stats:
                msg += f" | loadσ {moe_stats['moe/load_std_mean']:.3f}"
            print(msg)

            log_payload = {
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "train/lr": lr,
                "train/tokens_per_sec": tps,
                "train/grad_norm": grad_norm,
                "train/fp8_active": use_fp8_now,
                "sys/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
                "iter": iter_num,
            }
            log_payload.update(moe_stats)
            logger.log_metrics(log_payload, step=iter_num)

            tokens_processed = 0
            t0 = time.time()

        # Eval
        if iter_num % tcfg.eval_interval == 0:
            val_loss = evaluate(model, loader, tcfg, fp8_recipe)
            # Normalize MTP loss for correct perplexity
            mtp_weight = 2.75 if mcfg.use_mtp else 1.0
            val_ppl = math.exp(min(20.0, val_loss / mtp_weight))
            print(f"[eval] it {iter_num}: val {val_loss:.4f} | ppl {val_ppl:.2f}")
            logger.log_metrics({"val/loss": val_loss, "val/ppl": val_ppl}, step=iter_num)

            if val_loss < best_val:
                best_val = val_loss
                logger.set_summary(best_val=best_val, best_iter=iter_num)
                save_checkpoint(model, optimizer, tcfg, iter_num, val_loss,
                                Path(tcfg.checkpoint_dir) / "best.pt")

        # Regular checkpoints
        if iter_num > 0 and iter_num % tcfg.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, tcfg, iter_num, float("nan"),
                            Path(tcfg.checkpoint_dir) / f"ckpt_{iter_num}.pt")

    # Final eval
    final_val = evaluate(model, loader, tcfg, fp8_recipe)
    mtp_weight = 2.75 if mcfg.use_mtp else 1.0
    final_ppl = math.exp(min(20.0, final_val / mtp_weight))
    logger.set_summary(final_val=final_val, final_ppl=final_ppl, total_iters=tcfg.max_iters, params=model.num_parameters())
    print("""
==============================
Training complete
==============================
Best val loss:  {:.4f}
Final val loss: {:.4f}
Final ppl:      {:.2f}
Params:         {:.1f}M
""".format(best_val, final_val, final_ppl, model.num_parameters()/1e6))
    logger.finish()


if __name__ == "__main__":
    main()
