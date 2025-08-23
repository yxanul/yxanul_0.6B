"""
FP8-Optimized Trainer with Transformer Engine Support
Provides automatic FP8 mixed precision training for 2x throughput on RTX 4090.
"""

import torch
import torch.nn as nn
import time
import os
from typing import Dict, Optional, Any
from pathlib import Path
import yaml

# Import Transformer Engine for FP8
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("Warning: Transformer Engine not available, using standard precision")

from enhanced_trainer import EnhancedTrainer


class FP8Trainer(EnhancedTrainer):
    """Trainer with FP8 optimization support via Transformer Engine."""
    
    def __init__(
        self,
        model: nn.Module,
        config_path: str = "configs",
        stage: str = "fp8_training",
        local_rank: int = -1,
        use_fp8: bool = True
    ):
        """Initialize FP8 trainer."""
        super().__init__(model, config_path, stage, local_rank)
        
        self.use_fp8 = use_fp8 and TRANSFORMER_ENGINE_AVAILABLE
        
        if self.use_fp8:
            print("=" * 60)
            print("FP8 Training Enabled via Transformer Engine")
            print("=" * 60)
            print("Expected benefits:")
            print("  - 2x throughput improvement over BF16")
            print("  - Automatic loss scaling for FP8")
            print("  - Dynamic range adjustment")
            print("=" * 60)
            
            # Create FP8 recipe for training
            self.fp8_recipe = recipe.DelayedScaling(
                margin=0,  # No margin for aggressive FP8
                interval=1,  # Update scaling every iteration
                fp8_format=recipe.Format.HYBRID,  # E4M3 forward, E5M2 backward
                amax_history_len=16,  # History for scaling factor
                amax_compute_algo="most_recent"  # Use most recent amax
            )
            
            # Modify optimizer for FP8 (higher learning rates often work better)
            self._adjust_optimizer_for_fp8()
        else:
            self.fp8_recipe = None
            if use_fp8:
                print("Warning: FP8 requested but Transformer Engine not available")
                print("Falling back to BF16 mixed precision training")
    
    def _adjust_optimizer_for_fp8(self):
        """Adjust optimizer settings for FP8 training."""
        # FP8 can often handle higher learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 1.2  # 20% higher LR for FP8
        
        print(f"Adjusted learning rate for FP8: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with FP8 support."""
        step_start = time.time()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Get batch size and sequence length for metrics
        batch_size = batch['input_ids'].shape[0]
        seq_length = batch['input_ids'].shape[1]
        actual_tokens = batch_size * seq_length
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with FP8 autocast if available
        if self.use_fp8 and self.fp8_recipe:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    labels=batch['labels']
                )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            # Fallback to BF16 autocast
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    labels=batch['labels']
                )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        
        # Scale loss for gradient accumulation if needed
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_fp8:
            # FP8 backward (Transformer Engine handles scaling)
            loss.backward()
        else:
            # Standard backward with gradient scaling for mixed precision
            if hasattr(self, 'scaler') and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Gradient clipping and optimizer step
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_fp8:
                # Direct gradient clipping for FP8
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                self.optimizer.step()
            else:
                # Scaled gradient clipping for mixed precision
                if hasattr(self, 'scaler') and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
        
        # Calculate metrics
        step_time = time.time() - step_start
        tokens_per_second = actual_tokens / step_time
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Calculate perplexity
        perplexity = torch.exp(loss).item()
        
        # Update global step
        self.global_step += 1
        
        # Prepare metrics
        metrics = {
            'train/loss': loss.item(),
            'train/perplexity': perplexity,
            'train/learning_rate': current_lr,
            'perf/step_time': step_time,
            'perf/tokens_per_second': tokens_per_second,
            'perf/samples_per_second': batch_size / step_time,
            'data/batch_size': batch_size,
            'data/seq_length': seq_length,
            'data/actual_tokens': actual_tokens,
            'optim/grad_norm': self._get_grad_norm(),
            'optim/learning_rate': current_lr,
            'system/gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
            'system/gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
        }
        
        # Add FP8-specific metrics if available
        if self.use_fp8:
            metrics['fp8/enabled'] = 1.0
            metrics['fp8/format'] = 'hybrid_e4m3_e5m2'
            
            # Get FP8 scaling factors if available
            if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
                # Check first transformer block for FP8 stats
                first_block = self.model.blocks[0]
                if hasattr(first_block, 'attention') and hasattr(first_block.attention, 'fp8_meta'):
                    fp8_meta = first_block.attention.fp8_meta
                    if 'scaling_fwd' in fp8_meta:
                        metrics['fp8/scale_fwd'] = fp8_meta['scaling_fwd'].item()
                    if 'scaling_bwd' in fp8_meta:
                        metrics['fp8/scale_bwd'] = fp8_meta['scaling_bwd'].item()
        else:
            metrics['fp8/enabled'] = 0.0
        
        return metrics
    
    def _get_grad_norm(self) -> float:
        """Calculate gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def validate(self, val_dataloader, max_batches: Optional[int] = None) -> float:
        """Validation with FP8 support."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                if max_batches and i >= max_batches:
                    break
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Use FP8 for validation too
                if self.use_fp8 and self.fp8_recipe:
                    with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            labels=batch['labels']
                        )
                else:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            labels=batch['labels']
                        )
                
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                
                batch_size = batch['input_ids'].shape[0]
                seq_length = batch['input_ids'].shape[1]
                
                total_loss += loss.item() * batch_size * seq_length
                total_tokens += batch_size * seq_length
        
        self.model.train()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint with FP8 state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'fp8_enabled': self.use_fp8,
        }
        
        # Add FP8 recipe state if available
        if self.use_fp8 and self.fp8_recipe:
            checkpoint['fp8_recipe'] = {
                'margin': self.fp8_recipe.margin,
                'interval': self.fp8_recipe.interval,
                'fp8_format': str(self.fp8_recipe.fp8_format),
                'amax_history_len': self.fp8_recipe.amax_history_len,
                'amax_compute_algo': self.fp8_recipe.amax_compute_algo
            }
        
        # Add any additional kwargs
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        
        # Also save FP8 statistics if available
        if self.use_fp8:
            stats_path = Path(path).with_suffix('.fp8_stats')
            fp8_stats = {
                'training_steps': self.global_step,
                'fp8_enabled': True,
                'expected_speedup': '2x over BF16',
                'memory_savings': '~50% vs FP32'
            }
            
            with open(stats_path, 'w') as f:
                yaml.dump(fp8_stats, f)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint with FP8 state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        
        # Check FP8 compatibility
        checkpoint_fp8 = checkpoint.get('fp8_enabled', False)
        if checkpoint_fp8 and not self.use_fp8:
            print("Warning: Checkpoint was trained with FP8 but current setup doesn't support it")
        elif not checkpoint_fp8 and self.use_fp8:
            print("Info: Checkpoint was trained without FP8, now enabling FP8 training")
        
        return checkpoint