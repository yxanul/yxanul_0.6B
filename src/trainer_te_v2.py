#!/usr/bin/env python3
"""
Trainer for TransformerEngine v2.4 Models

Key differences from old trainer:
1. Proper fp8_autocast usage (backward outside context)
2. Native support for TE modules
3. Optimized FP8 recipes for H100
4. Better gradient accumulation with FP8
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import numpy as np

# Import base trainer for common functionality
from enhanced_trainer import EnhancedTrainer

# TransformerEngine v2.4 imports
try:
    import transformer_engine as te
    import transformer_engine.pytorch as te_pytorch
    from transformer_engine.common.recipe import (
        DelayedScaling, 
        Format
    )
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("WARNING: TransformerEngine not available. Install from NGC container.")


class TEv2Trainer(EnhancedTrainer):
    """
    Trainer specifically optimized for TransformerEngine v2.4 models.
    
    Inherits monitoring and logging from EnhancedTrainer but overrides
    training step to properly use fp8_autocast according to v2.4 best practices.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config_path: str = "configs",
        stage: str = "te_v2_training",
        local_rank: int = -1,
        use_fp8: bool = True,
        fp8_format: str = "hybrid",  # "hybrid", "e4m3", or "mxfp8"
        fp8_recipe=None,  # Optionally pass pre-created recipe
        calibration_steps: int = 64,  # More calibration for FP8 stability
        gradient_accumulation_steps: int = 32,
        learning_rate: float = 2e-4,  # Conservative for 270M model
        batch_size: int = 1,
        **kwargs
    ):
        # Initialize base trainer
        super().__init__(model, config_path, stage, local_rank, **kwargs)
        
        if not TE_AVAILABLE:
            print("WARNING: TransformerEngine not available. Falling back to BF16 training.")
            use_fp8 = False
        
        self.use_fp8 = use_fp8 and TE_AVAILABLE
        self.calibration_steps = calibration_steps
        self.fp8_format = fp8_format
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)  # Default gradient clipping
        self.global_step = 0  # Initialize global step counter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Initialize scheduler with linear warmup + cosine decay
        self.warmup_steps = kwargs.get('warmup_steps', 1000)
        # Calculate proper num_training_steps based on dataset
        # For 1B tokens with batch_size=1, seq_len=2048, grad_accum=32:
        # tokens_per_step = 1 * 2048 * 32 = 65536
        # total_steps = 1e9 / 65536 = ~15,258 steps
        tokens_per_step = self.batch_size * 2048 * self.gradient_accumulation_steps
        default_total_steps = int(1e9 / tokens_per_step)  # For 1B token dataset
        self.num_training_steps = kwargs.get('num_training_steps', default_total_steps)
        print(f"Training steps calculated: {self.num_training_steps} (for 1B tokens)")
        
        # Use linear warmup followed by cosine decay
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            """Linear warmup then cosine decay"""
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - self.warmup_steps) / float(max(1, self.num_training_steps - self.warmup_steps))
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        print(f"Scheduler: {self.warmup_steps} warmup steps, then cosine decay to 10% LR")
        
        # MXFP8 is only available on Blackwell GPUs
        if fp8_format == "mxfp8":
            print("WARNING: MXFP8 is only supported on Blackwell GPUs (B100, B200, GB200).")
            print("Falling back to hybrid format for H100/RTX 4090.")
            self.fp8_format = "hybrid"
        
        # Use provided recipe or create one based on format choice
        if self.use_fp8:
            if fp8_recipe is not None:
                # Use the provided recipe (ensures consistency with model init)
                self.fp8_recipe = fp8_recipe
                print(f"FP8 Training enabled with provided recipe ({fp8_format} format)")
            else:
                # Create new recipe if not provided
                self.fp8_recipe = self._create_fp8_recipe(fp8_format)
                print(f"FP8 Training enabled with {fp8_format} format")
            
            # Adjust optimizer for FP8 (often benefits from higher LR)
            self._adjust_optimizer_for_fp8()
        else:
            self.fp8_recipe = None
            print("Training in BF16 precision")
    
    def _create_fp8_recipe(self, format_type: str):
        """
        Create FP8 recipe based on format choice.
        
        TE v2.4 supports multiple FP8 formats:
        - hybrid: E4M3 forward, E5M2 backward (best for stability)
        - e4m3: E4M3 everywhere (more precision)
        
        Note: MXFP8 is only supported on Blackwell GPUs (B100, B200, GB200)
        """
        if format_type == "e4m3":
            # E4M3 everywhere
            print("Using E4M3 format")
            return DelayedScaling(
                fp8_format=Format.E4M3,
                amax_history_len=16,  # Shorter history per TE v2.4 docs
                amax_compute_algo="max",
                reduce_amax=True,  # Better for multi-GPU
                fp8_dpa=False,  # Disable FP8 for attention (avoids cuDNN kernel issues)
                fp8_mha=False   # Disable FP8 for multi-head attention
            )
        else:  # hybrid (default)
            # E4M3 forward, E5M2 backward
            print("Using hybrid format (E4M3 forward, E5M2 backward)")
            print("Note: FP8 disabled for attention layers (using BF16) to avoid cuDNN issues")
            return DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max",
                reduce_amax=True,
                fp8_dpa=False,  # Run attention in BF16 to avoid cuDNN FP8 kernel issues
                fp8_mha=False   # Run multi-head attention in BF16
            )
    
    def _adjust_optimizer_for_fp8(self):
        """Keep LR unchanged for FP8 - rely on warmup for stability"""
        print(f"FP8 enabled; keeping LR at {self.optimizer.param_groups[0]['lr']:.2e}")
        print("Using warmup scheduler for stable training start")
    
    def _compute_gradient_stats(self) -> Dict[str, float]:
        """Compute gradient statistics for monitoring."""
        try:
            total_norm = 0.0
            grad_values = []
            
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    grad_values.append(p.grad.data.abs().max().item())
            
            total_norm = total_norm ** 0.5
            
            if grad_values:
                return {
                    "total_norm": total_norm,
                    "max_grad": max(grad_values),
                    "min_grad": min(grad_values) if grad_values else 0,
                }
        except Exception:
            pass
        return None
    
    def _compute_weight_stats(self) -> Dict[str, float]:
        """Compute model weight statistics."""
        try:
            weight_values = []
            
            for name, p in self.model.named_parameters():
                if 'weight' in name and p.data is not None:
                    weight_values.extend(p.data.cpu().numpy().flatten())
            
            if weight_values:
                weight_array = np.array(weight_values)
                return {
                    "mean": float(np.mean(weight_array)),
                    "std": float(np.std(weight_array)),
                    "max": float(np.max(weight_array)),
                    "min": float(np.min(weight_array)),
                }
        except Exception:
            pass
        return None
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Training step following TE v2.4 best practices.
        
        Key difference: backward pass happens OUTSIDE fp8_autocast!
        """
        start_time = time.time()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Determine if this is a calibration step
        is_calibration = self.global_step < self.calibration_steps
        
        # Forward pass WITH fp8_autocast
        if self.use_fp8:
            with te_pytorch.fp8_autocast(
                enabled=True,
                fp8_recipe=self.fp8_recipe,
                calibrating=is_calibration  # First N steps calibrate scaling
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                logits = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
        else:
            # Regular BF16/FP32 forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                logits = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
        
        # Scale loss for gradient accumulation if needed
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass OUTSIDE fp8_autocast (TE v2.4 requirement!)
        # BF16 doesn't need GradScaler (unlike FP16), and FP8 handles its own scaling
        loss.backward()
        
        # Gradient accumulation and optimizer step
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        # Calculate metrics
        with torch.no_grad():
            actual_loss = (loss.detach() if loss.requires_grad else loss) * self.gradient_accumulation_steps
            perplexity = torch.exp(actual_loss).item()
            # Clamp perplexity to reasonable range
            perplexity = min(perplexity, 1e6)  # Cap at 1M to avoid overflow
            
            # Calculate accuracy if we have logits
            accuracy = 0
            if logits is not None and labels is not None:
                shifted_labels = labels[..., 1:].contiguous()
                shifted_logits = logits[..., :-1, :].contiguous()
                preds = torch.argmax(shifted_logits, dim=-1)
                mask = shifted_labels != -100
                if mask.sum() > 0:
                    accuracy = ((preds == shifted_labels) & mask).float().sum() / mask.sum()
                    accuracy = accuracy.item()
        
        # Increment global step
        self.global_step += 1
        
        # Calculate throughput
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        step_time = time.time() - start_time
        tokens_per_sec = (batch_size * seq_len) / step_time
        
        metrics = {
            "loss": actual_loss.item(),
            "perplexity": perplexity,
            "accuracy": accuracy,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "tokens_per_second": tokens_per_sec,
            "step_time": step_time,
            "global_step": self.global_step,
        }
        
        # Enhanced WandB logging with comprehensive metrics
        try:
            import wandb
            if wandb.run is not None:
                # Basic training metrics
                log_dict = {
                    "train/loss": metrics["loss"],
                    "train/perplexity": metrics["perplexity"],
                    "train/accuracy": metrics["accuracy"],
                    "train/learning_rate": metrics["learning_rate"],
                    "train/tokens_per_second": metrics["tokens_per_second"],
                    "train/step_time": metrics["step_time"],
                }
                
                # GPU metrics
                if torch.cuda.is_available():
                    log_dict.update({
                        "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                        "gpu/memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                        "gpu/memory_utilization": (torch.cuda.memory_allocated() / torch.cuda.memory_reserved()) * 100 if torch.cuda.memory_reserved() > 0 else 0,
                        "gpu/max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    })
                    
                    # Try to get GPU utilization (may not be available on all systems)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        log_dict.update({
                            "gpu/utilization": gpu_util.gpu,
                            "gpu/memory_utilization_percent": gpu_util.memory,
                        })
                        
                        # Temperature monitoring
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        log_dict["gpu/temperature"] = temp
                        
                        # Power draw
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        log_dict["gpu/power_watts"] = power
                    except:
                        pass  # NVML not available
                
                # Gradient statistics (every 100 steps to avoid overhead)
                if self.global_step % 100 == 0:
                    grad_stats = self._compute_gradient_stats()
                    if grad_stats:
                        log_dict.update({
                            "gradients/norm": grad_stats["total_norm"],
                            "gradients/max": grad_stats["max_grad"],
                            "gradients/min": grad_stats["min_grad"],
                        })
                
                # Model weight statistics (every 500 steps)
                if self.global_step % 500 == 0:
                    weight_stats = self._compute_weight_stats()
                    if weight_stats:
                        log_dict.update({
                            "weights/mean": weight_stats["mean"],
                            "weights/std": weight_stats["std"],
                            "weights/max": weight_stats["max"],
                            "weights/min": weight_stats["min"],
                        })
                
                # FP8 specific metrics
                if self.use_fp8:
                    log_dict.update({
                        "fp8/calibrating": is_calibration,
                        "fp8/calibration_progress": min(1.0, self.global_step / self.calibration_steps),
                    })
                    
                    # Check for FP8 scale factors if available
                    if hasattr(self.model, 'get_fp8_scales'):
                        fp8_scales = self.model.get_fp8_scales()
                        for name, scale in fp8_scales.items():
                            log_dict[f"fp8/scale_{name}"] = scale
                
                # Batch and sequence statistics
                log_dict.update({
                    "batch/size": batch_size,
                    "batch/seq_length": seq_len,
                    "batch/total_tokens": batch_size * seq_len,
                    "batch/gradient_accumulation": self.gradient_accumulation_steps,
                    "batch/effective_batch_size": batch_size * self.gradient_accumulation_steps,
                })
                
                # Training progress
                if hasattr(self, 'num_training_steps'):
                    log_dict["progress/percentage"] = (self.global_step / self.num_training_steps) * 100
                
                wandb.log(log_dict, step=self.global_step)
        except Exception as e:
            # Only print error once every 1000 steps to avoid spam
            if self.global_step % 1000 == 0:
                print(f"WandB logging error at step {self.global_step}: {e}")
        
        # Add FP8-specific metrics
        if self.use_fp8:
            metrics["fp8_calibration"] = is_calibration
            if hasattr(self.model, 'get_fp8_metrics'):
                metrics.update(self.model.get_fp8_metrics())
        
        return metrics
    
    def validate(self, val_dataloader, max_batches: Optional[int] = None) -> float:
        """
        Validation with proper TE v2.4 fp8_autocast usage.
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                if max_batches and i >= max_batches:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass with FP8 if enabled
                if self.use_fp8:
                    with te_pytorch.fp8_autocast(
                        enabled=True,
                        fp8_recipe=self.fp8_recipe,
                        calibrating=False  # No calibration during validation
                    ):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Accumulate loss
                batch_tokens = attention_mask.sum().item() if attention_mask is not None else input_ids.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        self.model.train()
        return perplexity
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics including FP8-specific info"""
        stats = super().get_training_stats()
        
        if self.use_fp8:
            stats["fp8"] = {
                "enabled": True,
                "format": self.fp8_format,
                "calibration_steps": self.calibration_steps,
                "recipe_type": type(self.fp8_recipe).__name__
            }
        else:
            stats["fp8"] = {"enabled": False}
        
        return stats


def create_te_v2_trainer(
    model: nn.Module,
    train_dataloader,
    val_dataloader=None,
    config: Optional[Dict] = None,
    use_fp8: bool = True,
    fp8_format: str = "hybrid"
) -> TEv2Trainer:
    """
    Factory function to create a TE v2.4 trainer.
    
    Args:
        model: The TE v2.4 model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config: Training configuration
        use_fp8: Whether to use FP8 training
        fp8_format: FP8 format ("hybrid", "e4m3", or "mxfp8")
    
    Returns:
        Configured TEv2Trainer instance
    """
    trainer = TEv2Trainer(
        model=model,
        config_path=config.get("config_path", "configs") if config else "configs",
        stage=config.get("stage", "te_v2_training") if config else "te_v2_training",
        use_fp8=use_fp8,
        fp8_format=fp8_format
    )
    
    # Set dataloaders
    trainer.train_dataloader = train_dataloader
    trainer.val_dataloader = val_dataloader
    
    return trainer


if __name__ == "__main__":
    """Test the trainer with a dummy model"""
    print("Testing TEv2Trainer...")
    
    if not TE_AVAILABLE:
        print("TransformerEngine not available. Skipping test.")
    else:
        # Create a dummy model
        from model_te_v2 import create_te_v2_model, ModelConfig
        
        config = ModelConfig(num_hidden_layers=2)  # Small for testing
        model = create_te_v2_model(config)
        
        # Create trainer
        trainer = TEv2Trainer(
            model=model,
            use_fp8=True,
            fp8_format="hybrid"
        )
        
        print(f"✓ TEv2Trainer created successfully")
        print(f"  FP8 enabled: {trainer.use_fp8}")
        print(f"  FP8 format: {trainer.fp8_format}")
        print(f"  Device: {trainer.device}")