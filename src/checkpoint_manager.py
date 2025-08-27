#!/usr/bin/env python3
"""
Checkpoint management system for efficient model saving and recovery.
Maintains a rolling window of checkpoints and tracks the best model.
"""

import os
import glob
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import torch


class CheckpointManager:
    """Manages checkpoint saving, rotation, and best model tracking."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_best: bool = True,
        metric_name: str = "val/perplexity",
        metric_mode: str = "min"  # "min" for loss/perplexity, "max" for accuracy
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        # Track best metric
        self.best_metric = float('inf') if metric_mode == "min" else float('-inf')
        self.best_checkpoint_path = None
        
        # Checkpoint history
        self.checkpoint_history = []
        self.load_checkpoint_history()
    
    def load_checkpoint_history(self):
        """Load existing checkpoint history from disk."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.checkpoint_history = data.get('checkpoints', [])
                self.best_metric = data.get('best_metric', self.best_metric)
                self.best_checkpoint_path = data.get('best_checkpoint', None)
    
    def save_checkpoint_history(self):
        """Save checkpoint history to disk."""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        data = {
            'checkpoints': self.checkpoint_history,
            'best_metric': self.best_metric,
            'best_checkpoint': self.best_checkpoint_path
        }
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int,
        metrics: Dict,
        config: Dict,
        scaler=None,
        fp8_recipe=None,
        curriculum_stage: Optional[int] = None,
        **extra_state
    ) -> str:
        """Save a checkpoint with automatic rotation and best model tracking."""
        
        # Generate checkpoint filename
        checkpoint_name = f"checkpoint_step{step:07d}_epoch{epoch:03d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': step,
            'metrics': metrics,
            'config': config,
            'curriculum_stage': curriculum_stage,
        }
        
        # Add optional components
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        if fp8_recipe is not None:
            # Handle DelayedScaling recipe from TransformerEngine
            checkpoint['fp8_recipe'] = {
                'margin': fp8_recipe.margin if hasattr(fp8_recipe, 'margin') else 0,
                'fp8_format': str(fp8_recipe.fp8_format) if hasattr(fp8_recipe, 'fp8_format') else 'HYBRID',
                'amax_history_len': fp8_recipe.amax_history_len if hasattr(fp8_recipe, 'amax_history_len') else 16,
                'amax_compute_algo': fp8_recipe.amax_compute_algo if hasattr(fp8_recipe, 'amax_compute_algo') else 'max',
                'reduce_amax': fp8_recipe.reduce_amax if hasattr(fp8_recipe, 'reduce_amax') else True,
                'fp8_dpa': fp8_recipe.fp8_dpa if hasattr(fp8_recipe, 'fp8_dpa') else False,
                'fp8_mha': fp8_recipe.fp8_mha if hasattr(fp8_recipe, 'fp8_mha') else False,
            }
        
        # Add any extra state
        checkpoint.update(extra_state)
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Update checkpoint history
        checkpoint_info = {
            'path': str(checkpoint_path),
            'step': step,
            'epoch': epoch,
            'metrics': metrics
        }
        self.checkpoint_history.append(checkpoint_info)
        
        # Handle best model
        if self.save_best and self.metric_name in metrics:
            current_metric = metrics[self.metric_name]
            is_better = (
                (self.metric_mode == "min" and current_metric < self.best_metric) or
                (self.metric_mode == "max" and current_metric > self.best_metric)
            )
            
            if is_better:
                self.best_metric = current_metric
                self.best_checkpoint_path = str(checkpoint_path)
                
                # Copy to best model file
                best_path = self.checkpoint_dir / "best_model.pt"
                shutil.copy2(checkpoint_path, best_path)
                print(f"  → New best model! {self.metric_name}={current_metric:.4f}")
                
                # Save best model info
                best_info_path = self.checkpoint_dir / "best_model_info.json"
                with open(best_info_path, 'w') as f:
                    json.dump({
                        'checkpoint': str(checkpoint_path),
                        'step': step,
                        'epoch': epoch,
                        'metric_name': self.metric_name,
                        'metric_value': current_metric,
                        'metrics': metrics
                    }, f, indent=2)
        
        # Rotate checkpoints (keep only the latest N)
        self._rotate_checkpoints()
        
        # Save history
        self.save_checkpoint_history()
        
        # Print checkpoint summary
        self._print_checkpoint_summary()
        
        return str(checkpoint_path)
    
    def _rotate_checkpoints(self):
        """Remove old checkpoints, keeping only the latest N."""
        # Get all checkpoint files (excluding best model and history)
        all_checkpoints = sorted(
            glob.glob(str(self.checkpoint_dir / "checkpoint_step*.pt")),
            key=lambda x: os.path.getmtime(x)
        )
        
        # Keep the best checkpoint regardless of age
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            if self.best_checkpoint_path in all_checkpoints:
                all_checkpoints.remove(self.best_checkpoint_path)
                checkpoints_to_check = all_checkpoints
                # We'll keep max_checkpoints - 1 regular checkpoints + the best one
                max_to_keep = self.max_checkpoints - 1
            else:
                checkpoints_to_check = all_checkpoints
                max_to_keep = self.max_checkpoints
        else:
            checkpoints_to_check = all_checkpoints
            max_to_keep = self.max_checkpoints
        
        # Remove old checkpoints
        if len(checkpoints_to_check) > max_to_keep:
            checkpoints_to_remove = checkpoints_to_check[:-max_to_keep]
            
            for checkpoint_path in checkpoints_to_remove:
                try:
                    os.remove(checkpoint_path)
                    print(f"  ✗ Removed old checkpoint: {Path(checkpoint_path).name}")
                    
                    # Also remove associated files (FP8 stats, etc.)
                    stats_file = Path(checkpoint_path).with_suffix('.fp8_stats')
                    if stats_file.exists():
                        os.remove(stats_file)
                    
                    # Remove from history
                    self.checkpoint_history = [
                        cp for cp in self.checkpoint_history 
                        if cp['path'] != checkpoint_path
                    ]
                except Exception as e:
                    print(f"  Warning: Failed to remove {checkpoint_path}: {e}")
    
    def _print_checkpoint_summary(self):
        """Print a summary of saved checkpoints."""
        active_checkpoints = [
            cp for cp in self.checkpoint_history 
            if os.path.exists(cp['path'])
        ]
        
        print(f"\n  Checkpoint Summary:")
        print(f"    Active checkpoints: {len(active_checkpoints)}/{self.max_checkpoints}")
        
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            print(f"    Best model: {Path(self.best_checkpoint_path).name}")
            print(f"    Best {self.metric_name}: {self.best_metric:.4f}")
        
        # Show checkpoint sizes
        total_size = 0
        for cp in active_checkpoints[-3:]:  # Show last 3
            if os.path.exists(cp['path']):
                size_mb = os.path.getsize(cp['path']) / (1024 * 1024)
                total_size += size_mb
                print(f"    • {Path(cp['path']).name}: {size_mb:.1f} MB")
        
        if len(active_checkpoints) > 3:
            print(f"    • ... and {len(active_checkpoints) - 3} more")
        
        print(f"    Total checkpoint size: {total_size:.1f} MB")
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        map_location: str = 'cuda'
    ) -> Dict:
        """Load a checkpoint."""
        if load_best and self.best_checkpoint_path:
            checkpoint_path = self.best_checkpoint_path
            print(f"Loading best checkpoint: {checkpoint_path}")
        elif checkpoint_path is None:
            # Load the latest checkpoint
            all_checkpoints = sorted(
                glob.glob(str(self.checkpoint_dir / "checkpoint_step*.pt")),
                key=lambda x: os.path.getmtime(x)
            )
            if all_checkpoints:
                checkpoint_path = all_checkpoints[-1]
                print(f"Loading latest checkpoint: {checkpoint_path}")
            else:
                raise ValueError("No checkpoints found to load")
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        print(f"✓ Loaded checkpoint from step {checkpoint['global_step']}")
        
        return checkpoint
    
    def get_resume_info(self) -> Optional[Dict]:
        """Get information for resuming training."""
        # Find the latest checkpoint
        all_checkpoints = sorted(
            glob.glob(str(self.checkpoint_dir / "checkpoint_step*.pt")),
            key=lambda x: os.path.getmtime(x)
        )
        
        if not all_checkpoints:
            return None
        
        latest_checkpoint = all_checkpoints[-1]
        
        # Load just the metadata (not the full model)
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        resume_info = {
            'checkpoint_path': latest_checkpoint,
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'curriculum_stage': checkpoint.get('curriculum_stage', None),
            'metrics': checkpoint.get('metrics', {})
        }
        
        print(f"Resume info: Step {resume_info['global_step']}, Epoch {resume_info['epoch']}")
        
        return resume_info


def integrate_checkpoint_manager(trainer_instance, config: Dict):
    """Helper to integrate CheckpointManager with existing trainer."""
    
    manager = CheckpointManager(
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        max_checkpoints=config.get('save_total_limit', 5),
        save_best=config.get('save_best_model', True),
        metric_name=config.get('metric_for_best_model', 'val/perplexity'),
        metric_mode='min' if 'perplexity' in config.get('metric_for_best_model', '') else 'max'
    )
    
    # Add to trainer
    trainer_instance.checkpoint_manager = manager
    
    return manager


if __name__ == "__main__":
    # Test the checkpoint manager
    print("Testing Checkpoint Manager")
    print("="*60)
    
    manager = CheckpointManager(
        checkpoint_dir="test_checkpoints",
        max_checkpoints=3,
        save_best=True
    )
    
    # Simulate saving checkpoints
    class DummyModel:
        def state_dict(self):
            return {'dummy': 'state'}
    
    model = DummyModel()
    optimizer = DummyModel()
    scheduler = DummyModel()
    
    for step in range(0, 50000, 10000):
        metrics = {
            'val/perplexity': 100 - step/1000,  # Improving metric
            'train/loss': 2.0 - step/10000
        }
        
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=step//10000,
            step=step,
            metrics=metrics,
            config={'test': True}
        )
    
    print("\nTest complete! Check test_checkpoints/ directory")