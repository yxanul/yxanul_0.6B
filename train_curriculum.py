#!/usr/bin/env python3
"""
Enhanced training script with full curriculum learning support.
Dynamically updates batch size, learning rate, and other parameters per stage.
"""

import os
import sys
import glob
import time
from pathlib import Path

# Disable WandB's automatic torch hooks before importing torch
os.environ["WANDB_DISABLE_SERVICE"] = "true"
os.environ["WANDB__REQUIRE_SERVICE"] = "false"

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import yaml
import argparse
from dataclasses import fields

# Add src to path
sys.path.append('src')

from model import create_model, ModelConfig
from enhanced_trainer import EnhancedTrainer
from data_pipeline import create_dataloader, create_tokenizer


class CurriculumManager:
    """Manages curriculum learning stage transitions."""
    
    def __init__(self, config):
        self.config = config
        self.stages = config.get('training', {}).get('curriculum_stages', [])
        self.current_stage_idx = 0
        self.current_stage = self.stages[0] if self.stages else None
        
    def get_stage_for_step(self, global_step):
        """Get the curriculum stage for the current step."""
        if not self.stages:
            return None
            
        for i, stage in enumerate(self.stages):
            if global_step >= stage.get('step', 0):
                self.current_stage_idx = i
                self.current_stage = stage
            else:
                break
                
        return self.current_stage
    
    def should_update_dataloader(self, global_step, last_batch_size):
        """Check if we need to recreate the dataloader."""
        stage = self.get_stage_for_step(global_step)
        if stage and stage.get('batch_size') != last_batch_size:
            return True, stage
        return False, stage
    
    def get_lr_scale(self, global_step):
        """Get learning rate scale for current stage."""
        stage = self.get_stage_for_step(global_step)
        return stage.get('lr_scale', 1.0) if stage else 1.0
    
    def get_grad_clip(self, global_step):
        """Get gradient clipping value for current stage."""
        stage = self.get_stage_for_step(global_step)
        return stage.get('grad_clip', 1.0) if stage else 1.0


def main():
    parser = argparse.ArgumentParser(description='Train Yxanul with Curriculum Learning')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file (e.g., configs/stage1_curriculum_optimized.yaml)')
    parser.add_argument('--model-config', type=str, default='configs/model_config.yaml',
                        help='Path to model config file')
    parser.add_argument('--optimization-config', type=str, default='configs/optimization.yaml',
                        help='Path to optimization config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation')
    args = parser.parse_args()
    
    # Verify config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file '{args.config}' not found!")
        print("Available configs:")
        for config in Path('configs').glob('*.yaml'):
            print(f"  - {config}")
        sys.exit(1)
    
    # Set device
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
        rank = 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    
    # Load configurations
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.optimization_config, 'r') as f:
        optimization_config = yaml.safe_load(f)
    
    # Combine configs
    full_config = {
        'model': model_config,
        'training': training_config.get('training', {}),
        'optimization': optimization_config,
        'data': training_config.get('data', {}),
        'validation': training_config.get('validation', {})
    }
    
    # Initialize curriculum manager
    curriculum_mgr = CurriculumManager(training_config)
    use_curriculum = training_config.get('training', {}).get('use_curriculum', False)
    
    if rank == 0:
        print("=" * 60)
        print("Yxanul 177M Training with Curriculum Learning")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Device: {device}")
        print(f"Curriculum Learning: {'ENABLED' if use_curriculum else 'DISABLED'}")
        if use_curriculum:
            print(f"Curriculum Stages: {len(curriculum_mgr.stages)}")
            for i, stage in enumerate(curriculum_mgr.stages):
                print(f"  Stage {i+1}: step={stage.get('step', 0)}, "
                      f"seq_len={stage.get('seq_len')}, "
                      f"batch_size={stage.get('batch_size')}, "
                      f"lr_scale={stage.get('lr_scale', 1.0)}")
        print("=" * 60)
    
    # Create model
    valid_fields = {f.name for f in fields(ModelConfig)}
    filtered_model_config = {k: v for k, v in model_config["model"].items() if k in valid_fields}
    model = create_model(filtered_model_config)
    model = model.to(device)
    
    # Create tokenizer - Using SuperBPE for 31% token reduction!
    tokenizer = create_tokenizer()  # Defaults to SuperBPE t=180k
    
    # Get initial batch size (from config or first curriculum stage)
    if use_curriculum and curriculum_mgr.stages:
        initial_batch_size = curriculum_mgr.stages[0].get('batch_size', 32)
        print(f"Initial batch size from curriculum: {initial_batch_size}")
    else:
        initial_batch_size = training_config.get('training', {}).get('per_device_train_batch_size', 32)
    
    # Create initial dataloaders
    train_dataloader, train_dataset = create_dataloader(
        dataset_name=training_config.get('data', {}).get('dataset_name', 'Yxanul/wikipedia-2k-high-quality'),
        tokenizer=tokenizer,
        batch_size=initial_batch_size,
        max_length=training_config.get('data', {}).get('max_sequence_length', 2048),
        stage_config=training_config,
        num_workers=2,
        split='train'
    )
    
    # Validation dataloader
    val_dataloader = None
    val_split = training_config.get('validation', {}).get('validation_split', 0.05)
    if val_split > 0:
        val_batch_size = training_config.get('validation', {}).get('per_device_eval_batch_size', 16)
        val_dataloader, _ = create_dataloader(
            dataset_name=training_config.get('data', {}).get('dataset_name'),
            tokenizer=tokenizer,
            batch_size=val_batch_size,
            max_length=training_config.get('data', {}).get('max_sequence_length', 2048),
            stage_config=training_config,
            num_workers=2,
            split=f'train[{int((1-val_split)*100)}%:]'
        )
    
    # Create trainer
    if rank == 0:
        print("\nInitializing Enhanced Trainer...")
    
    trainer = EnhancedTrainer(
        model=model,
        config_path="configs",
        stage="curriculum_training",
        local_rank=args.local_rank
    )
    
    # Set trainer attributes
    trainer.train_dataloader = train_dataloader
    trainer.val_dataloader = val_dataloader
    trainer.tokenizer = tokenizer
    trainer.device = device
    trainer.world_size = world_size
    trainer.local_rank = rank if rank >= 0 else 0
    trainer.config = full_config
    
    # Load checkpoint if provided
    start_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        if rank == 0:
            print(f"\nLoading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint.get('global_step', 0)
        if rank == 0:
            print(f"Resumed from step {start_step}")
    
    # Training parameters
    num_epochs = training_config.get('training', {}).get('num_epochs', 1)
    max_steps = training_config.get('training', {}).get('max_steps', -1)
    checkpoint_steps = training_config.get('training', {}).get('save_steps', 10000)
    eval_steps = training_config.get('training', {}).get('eval_steps', 2000)
    logging_steps = training_config.get('training', {}).get('logging_steps', 100)
    save_total_limit = training_config.get('training', {}).get('save_total_limit', 3)
    
    # Get base learning rate
    base_lr = training_config.get('training', {}).get('learning_rate', 6e-4)
    
    # Initialize WandB
    if rank == 0:
        trainer._init_wandb()
    
    # Evaluation only mode
    if args.eval_only:
        if rank == 0:
            print("\nRunning evaluation only...")
            if val_dataloader:
                val_ppl = trainer.validate(val_dataloader)
                print(f"Validation perplexity: {val_ppl:.2f}")
            else:
                print("No validation dataloader available")
        return
    
    # Training loop
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Steps: {max_steps if max_steps > 0 else 'Until epochs complete'}")
        print(f"Initial batch size: {initial_batch_size}")
        print(f"Base learning rate: {base_lr}")
        print("=" * 60)
    
    global_step = start_step
    current_batch_size = initial_batch_size
    last_curriculum_stage = -1
    
    # Main training loop
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
        
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        epoch_steps = 0
        
        # Create iterator with proper resumption
        train_iter = iter(train_dataloader)
        
        try:
            while True:
                # Check if we need to update dataloader for curriculum
                if use_curriculum:
                    should_update, stage = curriculum_mgr.should_update_dataloader(global_step, current_batch_size)
                    
                    if should_update:
                        new_batch_size = stage.get('batch_size', current_batch_size)
                        new_seq_len = stage.get('seq_len')
                        
                        if rank == 0:
                            print(f"\n[CURRICULUM UPDATE at step {global_step}]")
                            print(f"  Batch size: {current_batch_size} → {new_batch_size}")
                            print(f"  Sequence length: → {new_seq_len}")
                            print(f"  LR scale: {stage.get('lr_scale', 1.0)}x")
                            print(f"  Gradient clip: {stage.get('grad_clip', 1.0)}")
                        
                        # Update dataset sequence length
                        if hasattr(train_dataset, 'current_seq_length'):
                            train_dataset.current_seq_length = new_seq_len
                        
                        # Recreate dataloader with new batch size
                        train_dataloader, train_dataset = create_dataloader(
                            dataset_name=training_config.get('data', {}).get('dataset_name'),
                            tokenizer=tokenizer,
                            batch_size=new_batch_size,
                            max_length=training_config.get('data', {}).get('max_sequence_length', 2048),
                            stage_config=training_config,
                            num_workers=2,
                            split='train'
                        )
                        trainer.train_dataloader = train_dataloader
                        current_batch_size = new_batch_size
                        
                        # Update learning rate
                        lr_scale = stage.get('lr_scale', 1.0)
                        new_lr = base_lr * lr_scale
                        for param_group in trainer.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        
                        # Update gradient clipping
                        trainer.max_grad_norm = stage.get('grad_clip', 1.0)
                        
                        # Create new iterator
                        train_iter = iter(train_dataloader)
                
                # Get next batch
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
                
                # Training step
                metrics = trainer.train_step(batch)
                
                epoch_loss += metrics['train/loss']
                epoch_tokens += metrics.get('data/actual_tokens', 0)
                epoch_steps += 1
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0 and rank == 0:
                    print(f"  Step {global_step}: Loss={metrics['train/loss']:.4f}, "
                          f"PPL={metrics['train/perplexity']:.2f}, "
                          f"LR={metrics['optim/learning_rate']:.2e}, "
                          f"Tokens/s={metrics['perf/tokens_per_second']:.0f}")
                    
                    # Log curriculum info
                    if use_curriculum:
                        metrics['curriculum/stage'] = curriculum_mgr.current_stage_idx
                        metrics['curriculum/batch_size'] = current_batch_size
                        metrics['curriculum/seq_len'] = train_dataset.current_seq_length if hasattr(train_dataset, 'current_seq_length') else 0
                    
                    # Log to WandB
                    try:
                        import wandb
                        wandb.log(metrics, step=global_step)
                    except:
                        pass
                
                # Validation
                if val_dataloader is not None and global_step % eval_steps == 0:
                    if rank == 0:
                        print(f"\n  Running validation at step {global_step}...")
                    val_ppl = trainer.validate(val_dataloader, max_batches=50)
                    if rank == 0 and val_ppl:
                        print(f"  Validation perplexity: {val_ppl:.2f}")
                    model.train()
                
                # Checkpointing with cleanup
                if global_step % checkpoint_steps == 0 and rank == 0:
                    checkpoint_path = f"checkpoints/checkpoint_step{global_step}.pt"
                    os.makedirs("checkpoints", exist_ok=True)
                    
                    # Save checkpoint with curriculum info
                    checkpoint_data = {
                        'global_step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'scheduler_state_dict': trainer.scheduler.state_dict(),
                        'curriculum_stage': curriculum_mgr.current_stage_idx if use_curriculum else None,
                        'config': args.config
                    }
                    torch.save(checkpoint_data, checkpoint_path)
                    print(f"  Checkpoint saved to {checkpoint_path}")
                    
                    # Clean up old checkpoints
                    checkpoints = sorted(glob.glob("checkpoints/checkpoint_step*.pt"))
                    if len(checkpoints) > save_total_limit:
                        for old_checkpoint in checkpoints[:-save_total_limit]:
                            os.remove(old_checkpoint)
                            print(f"  Removed old checkpoint: {old_checkpoint}")
                
                # Check max steps
                if max_steps > 0 and global_step >= max_steps:
                    if rank == 0:
                        print(f"\nReached max steps ({max_steps})")
                    break
                    
        except KeyboardInterrupt:
            if rank == 0:
                print("\nTraining interrupted by user")
            break
        
        # Epoch summary
        if rank == 0 and epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            avg_tokens_per_second = epoch_tokens / epoch_steps
            
            print(f"\n[Epoch {epoch + 1} Complete]")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Total tokens: {epoch_tokens:,}")
            print(f"  Avg tokens/second: {avg_tokens_per_second:.0f}")
        
        # Check if we should stop
        if max_steps > 0 and global_step >= max_steps:
            break
    
    # Final checkpoint
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        final_checkpoint = f"checkpoints/final_model_step{global_step}.pt"
        checkpoint_data = {
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'curriculum_stage': curriculum_mgr.current_stage_idx if use_curriculum else None,
            'config': args.config
        }
        torch.save(checkpoint_data, final_checkpoint)
        print(f"\n[Training Complete] Final model saved to {final_checkpoint}")
        
        # Close WandB
        try:
            import wandb
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()