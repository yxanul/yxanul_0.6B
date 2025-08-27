#!/usr/bin/env python3
"""
Training script for Yxanul with TransformerEngine v2.4

This script uses the new TE v2.4 model and trainer for optimal performance on H100/A100.
Expected improvements:
- 40-50% faster training with native TE modules
- Better FP8 convergence with proper context usage
- Automatic Flash Attention 3 on NGC 25.05+
"""

import os
import sys
import time
import glob
import argparse
from pathlib import Path
from dataclasses import asdict

# Disable WandB service before importing torch
os.environ["WANDB_DISABLE_SERVICE"] = "true"
os.environ["WANDB_REQUIRE_SERVICE"] = "false"

import torch
import yaml

# Add src to path
sys.path.append('src')

# Import TE v2.4 components
from model_te_v2 import create_te_v2_model, ModelConfig
from trainer_te_v2 import TEv2Trainer
from data_pipeline import create_dataloader, create_tokenizer, create_curriculum_dataloader
from checkpoint_manager import CheckpointManager
from transformer_engine.common.recipe import DelayedScaling, Format

# Check for TransformerEngine
try:
    import transformer_engine as te
    print(f"TransformerEngine v{te.__version__} available")
    TE_AVAILABLE = True
except ImportError:
    print("ERROR: TransformerEngine not found!")
    print("Please use NVIDIA NGC container 25.05+ for optimal performance")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Yxanul with TE v2.4")
    
    # Model configuration
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Model configuration file')
    parser.add_argument('--model-size', type=str, default='197M',
                       choices=['197M', '270M'], help='Model size preset')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=6e-4,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--max-steps', type=int, default=-1,
                       help='Maximum training steps')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup steps')
    
    # FP8 configuration
    parser.add_argument('--fp8-format', type=str, default='hybrid',
                       choices=['hybrid', 'e4m3'],
                       help='FP8 format to use (MXFP8 requires Blackwell GPUs)')
    parser.add_argument('--no-fp8', action='store_true',
                       help='Disable FP8 training')
    parser.add_argument('--calibration-steps', type=int, default=10,
                       help='FP8 calibration steps')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_te_v2',
                       help='Checkpoint directory')
    parser.add_argument('--save-steps', type=int, default=10000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval-steps', type=int, default=2000,
                       help='Evaluate every N steps')
    parser.add_argument('--multi-domain-eval-steps', type=int, default=10000,
                       help='Run multi-domain validation every N steps')
    parser.add_argument('--save-total-limit', type=int, default=3,
                       help='Maximum checkpoints to keep')
    
    # Data
    parser.add_argument('--dataset', type=str, default='fineweb-edu-highest-quality-2025',
                       help='Dataset name')
    parser.add_argument('--max-length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')
    
    # Curriculum training
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum training with streaming')
    parser.add_argument('--curriculum-config', type=str, 
                       default='configs/yxanul_270m_progressive_curriculum.yaml',
                       help='Curriculum configuration file')
    parser.add_argument('--target-tokens', type=int, default=1_000_000_000,
                       help='Target tokens for curriculum training (1B default)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark mode')
    parser.add_argument('--local-rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    return parser.parse_args()


def load_config(args):
    """Load and merge configurations"""
    
    # Base model config with factorized embeddings
    if args.model_size == '197M':
        config = ModelConfig(
            vocab_size=200005,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=28,
            num_attention_heads=12,
            num_kv_heads=2,
            use_fp8=not args.no_fp8,
            use_factorized_embedding=True,
            factorization_dim=128
        )
    else:  # 270M
        config = ModelConfig(
            vocab_size=200005,
            hidden_size=896,
            intermediate_size=2400,
            num_hidden_layers=32,
            num_attention_heads=14,
            num_kv_heads=2,
            use_fp8=not args.no_fp8,
            use_factorized_embedding=True,
            factorization_dim=128
        )
    
    # Load from file if provided
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            # Update config with file values
            if 'model' in file_config:
                for key, value in file_config['model'].items():
                    if hasattr(config, key):
                        # Special handling for dtype strings
                        if key == 'params_dtype' and isinstance(value, str):
                            if value == 'bfloat16':
                                value = torch.bfloat16
                            elif value == 'float16':
                                value = torch.float16
                            elif value == 'float32':
                                value = torch.float32
                        setattr(config, key, value)
    
    return config


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("Yxanul Training with TransformerEngine v2.4")
    print("="*60)
    print(f"Model size: {args.model_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"FP8 format: {args.fp8_format if not args.no_fp8 else 'Disabled'}")
    if args.fp8_format == 'mxfp8':
        print("WARNING: MXFP8 requires Blackwell GPUs. Use 'hybrid' or 'e4m3' for H100/RTX 4090.")
    print(f"Max sequence length: {args.max_length}")
    print("="*60 + "\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name()
    print(f"Using GPU: {gpu_name}")
    
    # Verify H100/A100 for optimal performance
    if 'H100' in gpu_name:
        print("✓ H100 detected - optimal configuration")
    elif 'A100' in gpu_name:
        print("✓ A100 detected - good performance expected")
    else:
        print(f"⚠ {gpu_name} detected - may not achieve expected speedups")
    
    # Load configuration
    model_config = load_config(args)
    
    # Create FP8 recipe that will be used consistently
    fp8_recipe = None
    if model_config.use_fp8:
        # Create the recipe based on format choice
        if args.fp8_format == "e4m3":
            fp8_recipe = DelayedScaling(
                fp8_format=Format.E4M3,
                amax_history_len=16,
                amax_compute_algo="max",
                reduce_amax=True
            )
        else:  # hybrid (default)
            fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
                amax_history_len=16,
                amax_compute_algo="max",
                reduce_amax=True,
                fp8_dpa=True  # FP8 attention if supported
            )
        print(f"Created FP8 recipe: {args.fp8_format} format")
    
    # Create model with the same recipe
    print("\nCreating TE v2.4 model...")
    model = create_te_v2_model(model_config, fp8_recipe=fp8_recipe)
    
    # Create tokenizer
    print("Loading tokenizer...")
    tokenizer = create_tokenizer()
    
    # Create dataloaders
    if args.curriculum:
        print(f"Loading curriculum configuration from {args.curriculum_config}")
        with open(args.curriculum_config, 'r') as f:
            curriculum_config = yaml.safe_load(f)
        
        print(f"\nCurriculum Training Enabled:")
        print(f"  Target tokens: {args.target_tokens:,}")
        print(f"  Number of stages: {len(curriculum_config['training']['curriculum_stages'])}")
        print(f"  Streaming from HuggingFace: Yes")
        
        # Start with first stage
        current_stage = 0
        stage_config = curriculum_config['training']['curriculum_stages'][current_stage]
        
        print(f"\nStarting with Stage 1: {stage_config['name']}")
        print(f"  Sequence length: {stage_config['seq_len']}")
        print(f"  Batch size: {stage_config['batch_size']}")
        print(f"  Dataset mix: {stage_config['dataset_mix']}")
        
        train_dataloader, train_dataset = create_curriculum_dataloader(
            curriculum_config=curriculum_config,
            tokenizer=tokenizer,
            batch_size=stage_config['batch_size'],
            current_stage=current_stage,
            num_workers=0,  # Streaming works best with 0 workers
            split="train"
        )
        
        # For validation, use a simple split from FineWeb
        val_dataloader, val_dataset = create_dataloader(
            dataset_name="HuggingFaceFW/fineweb-edu",
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=stage_config['seq_len'],
            num_workers=0,
            split="train[95%:]"
        )
        
    else:
        print(f"Loading dataset: {args.dataset}")
        train_dataloader, train_dataset = create_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            split="train[:95%]"
        )
        
        val_dataloader, val_dataset = create_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=1,
            split="train[95%:]"
        )
        
        print(f"Train samples: {len(train_dataset):,}")
        print(f"Val samples: {len(val_dataset):,}")
    
    # Create trainer with the same FP8 recipe
    print("\nInitializing TE v2.4 Trainer...")
    trainer = TEv2Trainer(
        model=model,
        use_fp8=not args.no_fp8,
        fp8_format=args.fp8_format,
        fp8_recipe=fp8_recipe,  # Pass the same recipe
        calibration_steps=args.calibration_steps,
        local_rank=args.local_rank
    )
    
    # Set dataloaders
    trainer.train_dataloader = train_dataloader
    trainer.val_dataloader = val_dataloader
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints=args.save_total_limit,
        save_best=True,
        metric_name="val/perplexity",
        metric_mode="min"
    )
    trainer.checkpoint_manager = checkpoint_manager
    
    # Initialize WandB
    trainer._init_wandb()
    
    # Evaluation only mode
    if args.eval_only:
        print("\nRunning evaluation...")
        val_ppl = trainer.validate(val_dataloader, max_batches=100)
        print(f"Validation perplexity: {val_ppl:.2f}")
        return
    
    # Benchmark mode
    if args.benchmark:
        print("\nRunning benchmark (100 steps)...")
        benchmark_start = time.time()
        total_tokens = 0
        
        for step, batch in enumerate(train_dataloader):
            if step >= 100:
                break
            
            metrics = trainer.train_step(batch)
            total_tokens += args.batch_size * args.max_length
            
            if step % 10 == 0:
                print(f"Step {step}: {metrics['tokens_per_second']:.0f} tokens/sec")
        
        benchmark_time = time.time() - benchmark_start
        avg_throughput = total_tokens / benchmark_time
        
        print("\n" + "="*60)
        print("Benchmark Results")
        print("="*60)
        print(f"Total time: {benchmark_time:.2f} seconds")
        print(f"Average throughput: {avg_throughput:.0f} tokens/sec")
        print(f"Theoretical time for 1B tokens: {1e9 / avg_throughput / 3600:.2f} hours")
        
        # Compare with expected performance
        if 'H100' in gpu_name:
            expected = 150000  # Expected tokens/sec on H100 with TE v2.4
            efficiency = (avg_throughput / expected) * 100
            print(f"Efficiency vs expected: {efficiency:.1f}%")
        
        return
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    global_step = 0
    best_val_ppl = float('inf')
    
    # Curriculum tracking
    if args.curriculum:
        cumulative_tokens = 0
        current_stage_idx = 0
        stages = curriculum_config['training']['curriculum_stages']
        stage_tokens = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_start = time.time()
        epoch_loss = 0
        epoch_tokens = 0
        
        for step, batch in enumerate(train_dataloader):
            # Check if we need to transition to next curriculum stage
            if args.curriculum:
                batch_tokens = batch['input_ids'].shape[0] * batch['input_ids'].shape[1]
                cumulative_tokens += batch_tokens
                stage_tokens += batch_tokens
                
                # Check if we've reached target tokens
                if cumulative_tokens >= args.target_tokens:
                    print(f"\n{'='*60}")
                    print(f"REACHED TARGET: {args.target_tokens:,} tokens!")
                    print(f"{'='*60}")
                    break
                
                # Check if we need to transition to next stage
                current_stage = stages[current_stage_idx]
                if stage_tokens >= current_stage['tokens'] and current_stage_idx < len(stages) - 1:
                    current_stage_idx += 1
                    stage_tokens = 0
                    new_stage = stages[current_stage_idx]
                    
                    print(f"\n{'='*60}")
                    print(f"Transitioning to Stage {current_stage_idx + 1}: {new_stage['name']}")
                    print(f"  Sequence length: {new_stage['seq_len']}")
                    print(f"  Batch size: {new_stage['batch_size']}")
                    print(f"  Dataset mix: {new_stage['dataset_mix']}")
                    print(f"{'='*60}\n")
                    
                    # Update dataset stage
                    train_dataset.update_stage(current_stage_idx)
                    
                    # Recreate dataloader with new batch size
                    train_dataloader, _ = create_curriculum_dataloader(
                        curriculum_config=curriculum_config,
                        tokenizer=tokenizer,
                        batch_size=new_stage['batch_size'],
                        current_stage=current_stage_idx,
                        num_workers=0,
                        split="train"
                    )
                    
                    # Update validation dataloader sequence length
                    val_dataloader, _ = create_dataloader(
                        dataset_name="HuggingFaceFW/fineweb-edu",
                        tokenizer=tokenizer,
                        batch_size=args.batch_size,
                        max_length=new_stage['seq_len'],
                        num_workers=0,
                        split="train[95%:]"
                    )
                    
                    # Break inner loop to restart with new dataloader
                    break
            
            # Training step
            metrics = trainer.train_step(batch)
            
            epoch_loss += metrics['loss']
            epoch_tokens += args.batch_size * (args.max_length if not args.curriculum else stages[current_stage_idx]['seq_len'])
            global_step += 1
            
            # Logging
            if global_step % 100 == 0:
                log_msg = f"  Step {global_step}: Loss={metrics['loss']:.4f}, "
                log_msg += f"PPL={metrics['perplexity']:.2f}, "
                log_msg += f"Tokens/sec={metrics['tokens_per_second']:.0f}, "
                log_msg += f"LR={metrics['learning_rate']:.2e}"
                
                if args.curriculum:
                    progress_pct = (cumulative_tokens / args.target_tokens) * 100
                    log_msg += f" | Stage {current_stage_idx + 1}/{len(stages)} "
                    log_msg += f"({cumulative_tokens:,}/{args.target_tokens:,} tokens, {progress_pct:.1f}%)"
                
                print(log_msg)
            
            # Validation
            if global_step % args.eval_steps == 0:
                print(f"\nRunning validation at step {global_step}...")
                val_ppl = trainer.validate(val_dataloader, max_batches=50)
                print(f"Validation perplexity: {val_ppl:.2f}")
                
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    print(f"New best validation perplexity: {best_val_ppl:.2f}")
            
            # Multi-domain validation (less frequent, more comprehensive)
            if args.multi_domain_eval_steps > 0 and global_step % args.multi_domain_eval_steps == 0:
                print(f"\nRunning multi-domain validation at step {global_step}...")
                try:
                    # Use smaller batch count since validation sets are small
                    # This will use all of HumanEval (5 batches), most of GSM8K (20 batches), 
                    # and a good sample of C4 (20 batches)
                    domain_results = trainer.validate_multi_domain(max_batches=20)
                    # Domain results are already printed by the validator
                except Exception as e:
                    print(f"Multi-domain validation failed: {e}")
                    print("Continuing with regular training...")
            
            # Checkpointing
            if global_step % args.save_steps == 0:
                checkpoint_metrics = {
                    'train/loss': metrics['loss'],
                    'train/perplexity': metrics['perplexity'],
                    'val/perplexity': val_ppl if 'val_ppl' in locals() else None
                }
                
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    epoch=epoch,
                    step=global_step,
                    metrics=checkpoint_metrics,
                    config=asdict(model_config),
                    fp8_recipe=trainer.fp8_recipe if trainer.use_fp8 else None
                )
            
            # Check max steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"\nReached max steps ({args.max_steps})")
                break
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / (step + 1)
        print(f"\nEpoch {epoch + 1} completed:")
        print(f"  Time: {epoch_time/60:.2f} minutes")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Tokens processed: {epoch_tokens:,}")
        print(f"  Throughput: {epoch_tokens/epoch_time:.0f} tokens/sec")
        
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    
    # Final validation
    print("\nFinal validation...")
    final_val_ppl = trainer.validate(val_dataloader, max_batches=100)
    print(f"Final validation perplexity: {final_val_ppl:.2f}")
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    
    # Final multi-domain validation
    if args.multi_domain_eval_steps > 0:
        print("\nFinal multi-domain validation...")
        try:
            # Use all available data for final validation
            # This will use all batches from each domain (max 60 for C4)
            final_domain_results = trainer.validate_multi_domain(max_batches=100)
            print("\nMulti-domain validation complete. See summary above.")
        except Exception as e:
            print(f"Final multi-domain validation failed: {e}")
    
    # Save final checkpoint
    final_checkpoint = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        epoch=args.num_epochs,
        step=global_step,
        metrics={'val/perplexity': final_val_ppl},
        config=asdict(model_config),
        is_final=True
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total steps: {global_step}")
    print(f"Best validation PPL: {best_val_ppl:.2f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    
    # Print TE v2.4 specific summary
    if trainer.use_fp8:
        print("\nTE v2.4 FP8 Training Summary:")
        print(f"  FP8 format: {args.fp8_format}")
        print(f"  Calibration steps: {args.calibration_steps}")
        print("  Native modules used:")
        print("    - TransformerLayer with GQA")
        print("    - RMSNorm (native)")
        print("    - SwiGLU (native)")
        print("    - Flash Attention 3 (auto-selected)")
        print(f"\nNote: MXFP8 format is only available on Blackwell GPUs (B100/B200/GB200)")
        print(f"For H100/A100/RTX 4090, use 'hybrid' or 'e4m3' formats.")


if __name__ == "__main__":
    main()