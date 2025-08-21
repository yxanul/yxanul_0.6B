"""
Enhanced Trainer with Comprehensive WandB Monitoring
Extends the OptimizedTrainer with professional-grade metrics tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
from typing import Dict, Optional, Any
from tqdm import tqdm
import numpy as np

from trainer import OptimizedTrainer

class EnhancedTrainer(OptimizedTrainer):
    """Enhanced trainer with comprehensive monitoring and metrics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_stats = {}
        self.register_hooks()
    
    def register_hooks(self):
        """Register hooks for activation monitoring."""
        # Track key layer activations
        def hook_fn(name):
            def hook(module, input, output):
                if self.global_step % 500 == 0:  # Only track every 500 steps
                    if isinstance(output, torch.Tensor):
                        self.activation_stats[name] = {
                            'mean': output.detach().mean().item(),
                            'std': output.detach().std().item(),
                            'max': output.detach().max().item(),
                            'min': output.detach().min().item(),
                        }
            return hook
        
        # Register hooks for important layers
        if hasattr(self.model, 'model'):
            # Hook first and last transformer layers
            if hasattr(self.model.model, 'layers'):
                if len(self.model.model.layers) > 0:
                    self.model.model.layers[0].register_forward_hook(hook_fn('layer_0'))
                    self.model.model.layers[-1].register_forward_hook(hook_fn('layer_last'))
    
    def _init_wandb(self):
        """Enhanced WandB initialization with custom charts."""
        try:
            import wandb
            
            # Model parameter count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Custom configuration
            config = {
                "model": {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "architecture": self.config.get("model", {}),
                },
                "optimization": self.config.get("optimization", {}),
                "training": self.config.get("training", {}),
                "hardware": {
                    "gpus": self.world_size,
                    "gpu_type": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                    "mixed_precision": self.config["optimization"]["mixed_precision"]["dtype"],
                    "cuda_available": torch.cuda.is_available(),
                }
            }
            
            # Initialize with custom settings
            project_name = self.config.get("training", {}).get("wandb", {}).get("project", "yxanul-177m")
            stage_name = self.config.get("training", {}).get("stage", {}).get("name", "unknown")
            
            wandb.init(
                project=project_name,
                name=f"train_{stage_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                config=config,
                tags=[
                    f"params_{total_params/1e6:.0f}M",
                    f"layers_{self.config.get('model', {}).get('model', {}).get('num_layers', 0)}",
                    "factorized_embeddings",
                    "gqa",
                    "swiglu",
                    "rmsnorm"
                ],
                save_code=True,
                notes=f"Training Yxanul {total_params/1e6:.1f}M with comprehensive monitoring"
            )
            
            # Define custom metrics
            wandb.define_metric("train/loss", summary="min")
            wandb.define_metric("val/perplexity", summary="min")
            wandb.define_metric("train/accuracy", summary="max")
            wandb.define_metric("perf/tokens_per_second", summary="mean")
            wandb.define_metric("memory/allocated_gb", summary="max")
            
            # Watch model with detailed gradients
            wandb.watch(self.model, log="all", log_freq=100, log_graph=True)
            
            print(f"[OK] WandB initialized with comprehensive monitoring")
            print(f"    Project: {project_name}")
            print(f"    Tracking {total_params/1e6:.1f}M parameters")
            
        except Exception as e:
            print(f"[WARNING] WandB initialization failed: {e}")
            print("    Training will continue without WandB logging")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with comprehensive monitoring."""
        start_time = time.time()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        
        # Pre-forward metrics
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Memory before forward (GPU only)
        mem_before = 0
        max_mem_before = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1e9
            max_mem_before = torch.cuda.max_memory_allocated() / 1e9
        
        # Forward pass with mixed precision
        forward_start = time.time()
        if self.use_amp and self.scaler is not None:
            with autocast(dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                logits = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
        
        forward_time = time.time() - forward_start
        
        # Compute additional metrics from logits
        accuracy = 0
        entropy = 0
        top5_confidence = 0
        perplexity = torch.exp(loss).item()
        
        if logits is not None:
            with torch.no_grad():
                # Accuracy
                if labels is not None:
                    shifted_labels = labels[..., 1:].contiguous()
                    shifted_logits = logits[..., :-1, :].contiguous()
                    _, predicted = torch.max(shifted_logits, dim=-1)
                    mask = shifted_labels != -100
                    if mask.any():
                        correct = (predicted == shifted_labels) & mask
                        accuracy = correct.float().sum() / mask.float().sum()
                        accuracy = accuracy.item()
                
                # Entropy of predictions (confidence measure)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean().item()
                
                # Top-5 confidence
                vocab_size = logits.shape[-1]
                top5_probs, _ = torch.topk(probs.mean(dim=1), k=min(5, vocab_size), dim=-1)
                top5_confidence = top5_probs.sum(-1).mean().item()
        
        # Backward pass
        backward_start = time.time()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        backward_time = time.time() - backward_start
        
        # Gradient statistics (before clipping)
        grad_stats = self._compute_gradient_stats()
        
        # Gradient clipping
        grad_norm = 0.0
        grad_config = self.config["optimization"]["gradients"]
        if grad_config.get("clip_type") == "norm":
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                grad_config.get("clip_value", 1.0)
            )
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
        
        # Optimizer step
        optimizer_start = time.time()
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        optimizer_time = time.time() - optimizer_start
        
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Post-step metrics
        step_time = time.time() - start_time
        
        # Token counting
        if hasattr(self.tokenizer, 'pad_token_id'):
            actual_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()
        else:
            actual_tokens = batch_size * seq_len
        
        tokens_processed = actual_tokens * self.world_size
        tokens_per_second = tokens_processed / step_time if step_time > 0 else 0
        
        # Memory after step (GPU only)
        mem_after = 0
        max_mem_after = 0
        mem_reserved = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1e9
            max_mem_after = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
        
        # Comprehensive metrics
        metrics = {
            # Loss metrics
            "train/loss": loss.item(),
            "train/perplexity": perplexity,
            "train/accuracy": accuracy,
            "train/entropy": entropy,
            "train/top5_confidence": top5_confidence,
            
            # Learning rate and optimization
            "optim/learning_rate": self.scheduler.get_last_lr()[0],
            "optim/gradient_norm": grad_norm,
            "optim/gradient_mean": grad_stats["mean"],
            "optim/gradient_std": grad_stats["std"],
            "optim/gradient_max": grad_stats["max"],
            "optim/gradient_min": grad_stats["min"],
            "optim/num_zeros": grad_stats["num_zeros"],
            "optim/num_infs": grad_stats["num_infs"],
            "optim/num_nans": grad_stats["num_nans"],
            
            # Performance metrics
            "perf/tokens_per_second": tokens_per_second,
            "perf/samples_per_second": batch_size / step_time if step_time > 0 else 0,
            "perf/forward_time": forward_time,
            "perf/backward_time": backward_time,
            "perf/optimizer_time": optimizer_time,
            "perf/total_step_time": step_time,
            
            # Memory metrics (GPU)
            "memory/allocated_gb": mem_after,
            "memory/max_allocated_gb": max_mem_after,
            "memory/reserved_gb": mem_reserved,
            "memory/forward_delta_gb": mem_after - mem_before,
            
            # Data metrics
            "data/batch_size": batch_size,
            "data/sequence_length": seq_len,
            "data/actual_tokens": actual_tokens,
            "data/padding_ratio": 1 - (actual_tokens / (batch_size * seq_len)),
            
            # Model state
            "model/num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model/num_trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        # Add weight statistics (every 100 steps)
        if self.global_step % 100 == 0:
            weight_stats = self._compute_weight_stats()
            metrics.update(weight_stats)
        
        # Add activation statistics (every 500 steps)
        if self.global_step % 500 == 0 and self.activation_stats:
            for name, stats in self.activation_stats.items():
                for stat_name, value in stats.items():
                    metrics[f"activations/{name}/{stat_name}"] = value
            self.activation_stats.clear()  # Clear after logging
        
        # Update global step
        self.global_step += 1
        
        return metrics
    
    def _compute_gradient_stats(self) -> Dict[str, float]:
        """Compute detailed gradient statistics."""
        all_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                all_grads.append(param.grad.data.flatten())
        
        if len(all_grads) > 0:
            all_grads = torch.cat(all_grads)
            return {
                "mean": all_grads.mean().item(),
                "std": all_grads.std().item(),
                "max": all_grads.max().item(),
                "min": all_grads.min().item(),
                "num_zeros": (all_grads == 0).sum().item(),
                "num_infs": torch.isinf(all_grads).sum().item(),
                "num_nans": torch.isnan(all_grads).sum().item(),
            }
        return {
            "mean": 0, "std": 0, "max": 0, "min": 0, 
            "num_zeros": 0, "num_infs": 0, "num_nans": 0
        }
    
    def _compute_weight_stats(self) -> Dict[str, float]:
        """Compute weight statistics for monitoring training health."""
        stats = {}
        
        # Global weight statistics
        all_weights = []
        for param in self.model.parameters():
            if param.requires_grad:
                all_weights.append(param.data.flatten())
        
        if all_weights:
            all_weights = torch.cat(all_weights)
            stats["weights/global_mean"] = all_weights.mean().item()
            stats["weights/global_std"] = all_weights.std().item()
            stats["weights/global_max"] = all_weights.max().item()
            stats["weights/global_min"] = all_weights.min().item()
        
        # Per-layer statistics (sample important layers)
        important_layers = [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.13",  # Middle layer
            "model.layers.27",  # Last layer
            "model.ln_f",
        ]
        
        for layer_name in important_layers:
            for name, param in self.model.named_parameters():
                if layer_name in name and "weight" in name and param.requires_grad:
                    layer_key = layer_name.replace(".", "_")
                    stats[f"weights/{layer_key}/mean"] = param.data.mean().item()
                    stats[f"weights/{layer_key}/std"] = param.data.std().item()
                    
                    # Check for dead neurons (weights all zero or very small)
                    if len(param.shape) >= 2:
                        # Check output neurons
                        neuron_norms = param.data.norm(dim=1)
                        dead_neurons = (neuron_norms < 1e-6).sum().item()
                        total_neurons = param.shape[0]
                        if total_neurons > 0:
                            stats[f"weights/{layer_key}/dead_ratio"] = dead_neurons / total_neurons
        
        return stats
    
    def validate(self, dataloader=None, max_batches: int = 100):
        """Enhanced validation with comprehensive metrics."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        all_losses = []
        all_perplexities = []
        
        if dataloader is None:
            if self.val_dataloader is None:
                return None
            dataloader = self.val_dataloader
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Validation", disable=self.local_rank != 0)):
                if i >= max_batches:
                    break
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                logits = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
                
                # Calculate accuracy
                if logits is not None and labels is not None:
                    shifted_labels = labels[..., 1:].contiguous()
                    shifted_logits = logits[..., :-1, :].contiguous()
                    _, predicted = torch.max(shifted_logits, dim=-1)
                    mask = shifted_labels != -100
                    if mask.any():
                        correct = (predicted == shifted_labels) & mask
                        total_correct += correct.float().sum().item()
                
                # Accumulate metrics
                num_tokens = (labels != -100).sum().item() if labels is not None else input_ids.numel()
                if num_tokens > 0:
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
                    all_losses.append(loss.item())
                    all_perplexities.append(torch.exp(loss).item())
        
        # Calculate final metrics
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)
            accuracy = total_correct / total_tokens if total_correct > 0 else 0
            
            # Calculate std dev
            loss_std = np.std(all_losses) if all_losses else 0
            perplexity_std = np.std(all_perplexities) if all_perplexities else 0
            
            # Log validation metrics
            val_metrics = {
                "val/loss": avg_loss,
                "val/perplexity": perplexity,
                "val/accuracy": accuracy,
                "val/loss_std": loss_std,
                "val/perplexity_std": perplexity_std,
                "val/num_samples": len(all_losses),
            }
            
            # Log to WandB if available
            if self.local_rank == 0:
                try:
                    import wandb
                    wandb.log(val_metrics)
                except:
                    pass
            
            print(f"\n[Validation] Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.4f}")
            
            return perplexity
        return None
    
    def log_epoch_summary(self, epoch: int, epoch_metrics: Dict):
        """Log comprehensive epoch summary to WandB."""
        if self.local_rank == 0:
            summary = {
                "epoch/number": epoch,
                "epoch/avg_loss": epoch_metrics.get("avg_loss", 0),
                "epoch/avg_tokens_per_second": epoch_metrics.get("avg_tokens_per_second", 0),
                "epoch/total_tokens": epoch_metrics.get("total_tokens", 0),
                "epoch/learning_rate_end": self.scheduler.get_last_lr()[0],
                "epoch/total_steps": self.global_step,
            }
            
            # Add validation metrics if available
            if "val_perplexity" in epoch_metrics:
                summary.update({
                    "epoch/val_perplexity": epoch_metrics["val_perplexity"],
                    "epoch/val_loss": epoch_metrics.get("val_loss", 0),
                    "epoch/train_val_gap": epoch_metrics.get("avg_loss", 0) - epoch_metrics.get("val_loss", 0),
                })
            
            try:
                import wandb
                wandb.log(summary)
                print(f"\n[Epoch {epoch}] Summary logged to WandB")
            except:
                pass
    
    def save_checkpoint(self, path: str, epoch: int = None, best_val_metric: float = None):
        """Save checkpoint with additional metadata."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_metric': best_val_metric,
            'config': self.config,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"[OK] Checkpoint saved to {path}")
        
        # Log to WandB
        if self.local_rank == 0:
            try:
                import wandb
                wandb.save(path)
            except:
                pass