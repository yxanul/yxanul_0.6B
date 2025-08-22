#!/usr/bin/env python3
"""
Main training script for Yxanul 177M model.
Supports single GPU (RTX 4090) and multi-GPU training.
"""

import os
import sys

# Disable WandB's automatic torch hooks before importing torch
os.environ["WANDB_DISABLE_SERVICE"] = "true"
os.environ["WANDB__REQUIRE_SERVICE"] = "false"

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress Dynamo errors for WandB compatibility
import yaml
import argparse
from pathlib import Path
from dataclasses import fields

# Add src to path
sys.path.append('src')

from model import create_model, ModelConfig
from enhanced_trainer import EnhancedTrainer
from data_pipeline import create_dataloader, create_tokenizer

def main():
    parser = argparse.ArgumentParser(description='Train Yxanul 177M Model')
    parser.add_argument('--config', type=str, default='configs/stage1_wikipedia.yaml',
                        help='Path to training config file')
    parser.add_argument('--model-config', type=str, default='configs/model_config.yaml',
                        help='Path to model config file')
    parser.add_argument('--optimization-config', type=str, default='configs/optimization.yaml',
                        help='Path to optimization config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--use-deepspeed', action='store_true',
                        help='Use DeepSpeed for training')
    args = parser.parse_args()
    
    # Set device
    if args.local_rank == -1:
        # Single GPU training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
        rank = 0
    else:
        # Distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    
    # Load configurations
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)
    
    with open(args.optimization_config, 'r') as f:
        optimization_config = yaml.safe_load(f)
    
    # Combine configs for trainer
    full_config = {
        'model': model_config,
        'training': training_config,
        'optimization': optimization_config,
    }
    
    # Create model
    if rank == 0:
        print("=" * 60)
        print("Yxanul 177M Training")
        print("=" * 60)
        print(f"\nDevice: {device}")
        print(f"World size: {world_size}")
        print(f"Loading model configuration...")
    
    # Filter config for model creation
    valid_fields = {f.name for f in fields(ModelConfig)}
    filtered_model_config = {k: v for k, v in model_config["model"].items() if k in valid_fields}
    
    # Create model
    model = create_model(filtered_model_config)
    model = model.to(device)
    
    # Setup for distributed training if needed
    if world_size > 1:
        if args.use_deepspeed:
            # DeepSpeed will handle model wrapping
            pass
        else:
            # Use DDP
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[args.local_rank])
    
    # Create tokenizer
    tokenizer = create_tokenizer("gpt2")
    
    # Create data loaders
    if rank == 0:
        print("\nCreating data loaders...")
    
    # Training dataloader
    train_dataloader, train_dataset = create_dataloader(
        dataset_name=training_config.get('data', {}).get('dataset_name', 'Yxanul/wikipedia-2k-high-quality'),
        tokenizer=tokenizer,
        batch_size=training_config.get('training', {}).get('per_device_train_batch_size', 32),
        max_length=training_config.get('data', {}).get('max_sequence_length', 2048),
        stage_config=training_config,
        num_workers=2,
        split='train'
    )
    
    # Validation dataloader (optional)
    val_dataloader = None
    val_split = training_config.get('validation', {}).get('validation_split', 0.05)
    if val_split > 0:
        val_dataloader, _ = create_dataloader(
            dataset_name=training_config.get('data', {}).get('dataset_name', 'Yxanul/wikipedia-2k-high-quality'),
            tokenizer=tokenizer,
            batch_size=training_config.get('validation', {}).get('per_device_eval_batch_size', 32),
            max_length=training_config.get('data', {}).get('max_sequence_length', 2048),
            stage_config=training_config,
            num_workers=2,
            split=f'train[{int((1-val_split)*100)}%:]'  # Last 5% for validation
        )
    
    # Create trainer
    if rank == 0:
        print("\nInitializing Enhanced Trainer...")
    
    trainer = EnhancedTrainer(
        model=model,
        config_path="configs",
        stage="stage1_rtx4090",
        local_rank=args.local_rank
    )
    
    # Override some trainer attributes
    trainer.train_dataloader = train_dataloader
    trainer.val_dataloader = val_dataloader
    trainer.tokenizer = tokenizer
    trainer.device = device
    trainer.world_size = world_size
    trainer.local_rank = rank if rank >= 0 else 0
    trainer.config = full_config
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        if rank == 0:
            print(f"\nLoading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, step {trainer.global_step}")
    else:
        start_epoch = 0
    
    # Training parameters
    num_epochs = training_config.get('training', {}).get('num_train_epochs', 1)
    checkpoint_steps = training_config.get('training', {}).get('save_steps', 5000)
    eval_steps = training_config.get('training', {}).get('eval_steps', 1000)
    logging_steps = training_config.get('training', {}).get('logging_steps', 10)
    
    # Initialize WandB
    if rank == 0:
        trainer._init_wandb()
    
    # Training loop
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size per device: {training_config.get('training', {}).get('per_device_train_batch_size', 32)}")
        print(f"Gradient accumulation: {training_config.get('training', {}).get('gradient_accumulation_steps', 1)}")
        print(f"Total batch size: {training_config.get('training', {}).get('per_device_train_batch_size', 32) * world_size}")
        print("=" * 60)
    
    # Main training loop
    global_step = trainer.global_step
    for epoch in range(start_epoch, num_epochs):
        if rank == 0:
            print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
        
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        epoch_steps = 0
        
        for step, batch in enumerate(train_dataloader):
            # Training step with comprehensive monitoring
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
            
            # Checkpointing
            if global_step % checkpoint_steps == 0 and rank == 0:
                checkpoint_path = f"checkpoints/checkpoint_epoch{epoch}_step{global_step}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                trainer.save_checkpoint(checkpoint_path, epoch=epoch)
                print(f"  Checkpoint saved to {checkpoint_path}")
            
            # Update sequence length for curriculum learning
            if hasattr(train_dataset, 'update_sequence_length'):
                new_seq_len = train_dataset.update_sequence_length(global_step, num_epochs * len(train_dataloader))
                if new_seq_len != train_dataset.current_seq_length and rank == 0:
                    print(f"  Updated sequence length to {new_seq_len}")
        
        # Epoch summary
        if rank == 0:
            avg_loss = epoch_loss / epoch_steps
            avg_tokens_per_second = epoch_tokens / (epoch_steps * metrics.get('perf/total_step_time', 1))
            
            epoch_metrics = {
                'avg_loss': avg_loss,
                'avg_tokens_per_second': avg_tokens_per_second,
                'total_tokens': epoch_tokens,
            }
            
            if val_dataloader is not None:
                val_ppl = trainer.validate(val_dataloader)
                if val_ppl:
                    epoch_metrics['val_perplexity'] = val_ppl
            
            trainer.log_epoch_summary(epoch + 1, epoch_metrics)
            
            print(f"\n[Epoch {epoch + 1} Complete]")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Total tokens: {epoch_tokens:,}")
            print(f"  Avg tokens/second: {avg_tokens_per_second:.0f}")
    
    # Final checkpoint
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        final_checkpoint = "checkpoints/final_model.pt"
        trainer.save_checkpoint(final_checkpoint, epoch=num_epochs)
        print(f"\n[Training Complete] Final model saved to {final_checkpoint}")
        
        # Close WandB
        try:
            import wandb
            wandb.finish()
        except:
            pass

if __name__ == "__main__":
    main()