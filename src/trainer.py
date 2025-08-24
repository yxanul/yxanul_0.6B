"""
Optimized Multi-GPU Trainer for Yxanul 0.6B
Implements all speed optimizations for 8x A100/H100 training.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.cuda.amp import autocast, GradScaler
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
from transformers import get_linear_schedule_with_warmup
import yaml
from pathlib import Path
from typing import Dict, Optional
import wandb
from tqdm import tqdm
import time
from data_pipeline import (
    create_dataloader, 
    create_tokenizer,
    calculate_training_steps,
    estimate_dataset_size
)

# Try to import optional optimizations
try:
    from apex import amp
    from apex.optimizers import FusedAdam
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    print("Warning: Apex not available, using standard Adam")

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    import transformer_engine.pytorch as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("Warning: Transformer Engine not available, FP8 training disabled")


class OptimizedTrainer:
    """Trainer with all optimizations for maximum speed."""
    
    def __init__(
        self,
        model: nn.Module,
        config_path: str,
        stage: str = "stage1",
        local_rank: int = -1
    ):
        self.model = model
        self.config = self._load_configs(config_path, stage)
        self.local_rank = local_rank if local_rank != -1 else int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Initialize tokenizer
        self.tokenizer = create_tokenizer("gpt2")
        
        # Setup distributed training
        self._setup_distributed()
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Initialize total_steps before scheduler setup
        self.total_steps = 0
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup mixed precision
        self._setup_mixed_precision()
        
        # Metrics tracking
        self.metrics = {
            "loss": [],
            "learning_rate": [],
            "gradient_norm": [],
            "tokens_per_second": [],
            "perplexity": []
        }
        
        # Data pipeline setup
        self.train_dataloader = None
        self.val_dataloader = None
        self.dataset_size = 0
        self.total_steps = 0
        
        # Initialize WandB if rank 0
        if self.local_rank == 0:
            self._init_wandb()
        
    def _load_configs(self, config_path: str, stage: str) -> Dict:
        """Load all configuration files."""
        config_dir = Path(config_path)
        
        # Load model config
        with open(config_dir / "model_config.yaml", 'r') as f:
            model_config = yaml.safe_load(f)
            
        # Load optimization config
        with open(config_dir / "optimization.yaml", 'r') as f:
            opt_config = yaml.safe_load(f)
            
        # Load stage-specific config
        stage_file = config_dir / f"{stage}.yaml"
        if stage_file.exists():
            with open(stage_file, 'r') as f:
                stage_config = yaml.safe_load(f)
        else:
            stage_config = {}
            
        return {
            "model": model_config,
            "optimization": opt_config,
            "training": stage_config
        }
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            
            # Get project name from config or use default
            project_name = self.config.get("training", {}).get("wandb_project", "yxanul-0.6b")
            
            # Initialize wandb
            wandb.init(
                project=project_name,
                name=f"{self.config.get('training', {}).get('stage', {}).get('name', 'training')}",
                config={
                    "model": self.config.get("model", {}),
                    "optimization": self.config.get("optimization", {}),
                    "training": self.config.get("training", {}),
                }
            )
            
            # Watch the model (skip if torch.compile is enabled)
            if not self.config.get('optimization', {}).get('torch_compile', {}).get('enabled', False):
                wandb.watch(self.model, log="gradients", log_freq=100)
            else:
                print("Skipping wandb.watch() due to torch.compile compatibility")
            print("WandB initialized successfully")
            
        except ImportError:
            print("WandB not installed. Skipping initialization.")
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
    
    def _setup_distributed(self):
        """Initialize distributed training."""
        if self.world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Rank {self.local_rank}: Initialized on {self.device}")
        
    def _apply_optimizations(self):
        """Apply all model optimizations."""
        # Handle missing optimization config
        opt_config = self.config.get("optimization", {
            "torch_compile": {"enabled": True, "mode": "default"},
            "memory": {"gradient_checkpointing": {"enabled": True}}
        })
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Torch compile (PyTorch 2.0+)
        if opt_config.get("torch_compile", {}).get("enabled", False):
            print("Applying torch.compile optimization...")
            self.model = torch.compile(
                self.model,
                mode=opt_config["torch_compile"].get("mode", "default"),
                fullgraph=opt_config["torch_compile"].get("fullgraph", False),
                backend=opt_config["torch_compile"].get("backend", "inductor")
            )
        
        # Setup distributed model
        if self.world_size > 1:
            if opt_config.get("distributed", {}).get("deepspeed", {}).get("enabled", False):
                # DeepSpeed will handle model wrapping
                pass
            elif opt_config.get("distributed", {}).get("fsdp", {}).get("enabled", False):
                # FSDP setup
                print("Using FSDP for distributed training...")
                mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                )
                
                self.model = FSDP(
                    self.model,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    mixed_precision=mixed_precision_policy,
                    device_id=self.local_rank,
                    use_orig_params=True,
                )
            else:
                # Standard DDP
                print("Using DDP for distributed training...")
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False
                )
        
        # Enable gradient checkpointing if configured
        if opt_config.get("memory", {}).get("gradient_checkpointing", {}).get("enabled", False):
            print("Enabling gradient checkpointing...")
            self.model.gradient_checkpointing_enable()
            
    def _setup_optimizer(self):
        """Setup optimizer with all optimizations."""
        # Handle both nested and flat config structures
        if "optimization" in self.config:
            opt_config = self.config.get("optimization", {}).get("optimizer", {})
        else:
            # Flat structure - use defaults
            opt_config = {
                "type": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.1,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        
        # Get training config
        if "training" in self.config:
            # Check if it's nested or flat
            if isinstance(self.config["training"], dict) and "training" in self.config["training"]:
                train_config = self.config["training"]["training"]
            else:
                train_config = self.config["training"]
        else:
            train_config = {}
        
        # Prepare parameters with weight decay
        no_decay = opt_config.get("no_decay_params", ["bias", "layer_norm", "layernorm"])
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": float(opt_config.get("weight_decay", 0.1)),
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Choose optimizer
        if opt_config.get("fused", False) and APEX_AVAILABLE:
            print("Using FusedAdam optimizer...")
            self.optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=float(train_config.get("learning_rate", 2e-4)),
                betas=tuple(opt_config.get("betas", [0.9, 0.95])),
                eps=float(opt_config.get("eps", 1e-8))
            )
        elif opt_config.get("use_8bit", False) and BNB_AVAILABLE:
            print("Using 8-bit Adam optimizer...")
            self.optimizer = bnb.optim.Adam8bit(
                optimizer_grouped_parameters,
                lr=float(train_config.get("learning_rate", 2e-4)),
                betas=tuple(opt_config.get("betas", [0.9, 0.95])),
                eps=float(opt_config.get("eps", 1e-8))
            )
        else:
            print("Using standard AdamW optimizer...")
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=float(train_config.get("learning_rate", 2e-4)),
                betas=tuple(opt_config.get("betas", [0.9, 0.95])),
                eps=float(opt_config.get("eps", 1e-8))
            )
            
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Handle both nested and flat config structures
        if "training" in self.config:
            # Check if it's nested or flat
            if isinstance(self.config["training"], dict) and "training" in self.config["training"]:
                train_config = self.config["training"]["training"]
            else:
                train_config = self.config["training"]
        else:
            train_config = {}
        
        # Calculate total steps based on actual dataset
        if self.total_steps == 0:
            # Will be set when dataloader is created
            self.total_steps = 100000  # Temporary default
        
        warmup_steps = int(train_config.get("warmup_steps", 2000))
        
        # Linear schedule with warmup and decay to zero
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=int(self.total_steps)
        )
        
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        # Handle both nested and flat config structures
        if "optimization" in self.config and "mixed_precision" in self.config.get("optimization", {}):
            mp_config = self.config["optimization"]["mixed_precision"]
        else:
            # Use defaults for FP8/BF16
            mp_config = {
                "enabled": True,
                "dtype": "bfloat16",
                "fp8_config": {
                    "enabled": False
                }
            }
        
        if mp_config.get("enabled", False):
            dtype = mp_config.get("dtype", "bfloat16")
            
            if dtype == "fp8" and torch.cuda.get_device_capability()[0] >= 9 and TE_AVAILABLE:
                print("Using FP8 training (H100/H200)")
                # Configure FP8 with Transformer Engine
                self.use_fp8 = True
                self.use_amp = False
                self.scaler = None
                
                # Wrap model with FP8 autocast
                fp8_config = mp_config.get("fp8_config", {})
                self.fp8_enabled = fp8_config.get("enabled", True)
                self.fp8_format = fp8_config.get("format", "e4m3")
                self.fp8_amax_history_len = int(fp8_config.get("amax_history_len", 1024))
                self.fp8_amax_compute_algo = fp8_config.get("amax_compute_algo", "most_recent")
                
                # Apply FP8 to model
                if self.fp8_enabled:
                    # Convert model layers to FP8
                    te.fp8.FP8GlobalStateManager.reset()
                    self.model = te.fp8_model_init(self.model)
                    
            elif dtype == "fp8" and not TE_AVAILABLE:
                print("Warning: FP8 requested but Transformer Engine not available, falling back to BF16")
                self.use_fp8 = False
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
                self.scaler = None
            elif dtype == "bfloat16":
                print("Using BF16 mixed precision")
                self.use_fp8 = False
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # BF16 doesn't need loss scaling
                # Convert model to bf16 for Flash Attention
                self.model = self.model.to(torch.bfloat16)
                print("Model converted to BF16 dtype")
            else:
                print("Using FP16 mixed precision")
                self.use_fp8 = False
                self.use_amp = True
                self.amp_dtype = torch.float16
                self.scaler = GradScaler()
                # Convert model to fp16 for Flash Attention
                self.model = self.model.to(torch.float16)
                print("Model converted to FP16 dtype")
        else:
            self.use_fp8 = False
            self.use_amp = False
            self.scaler = None
            
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with all optimizations."""
        start_time = time.time()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch.get("labels", input_ids)
        
        # Forward pass with mixed precision
        if self.use_fp8 and TE_AVAILABLE:
            # FP8 training with Transformer Engine
            with te.fp8_autocast(
                enabled=self.fp8_enabled,
                fp8_format=self.fp8_format,
                amax_history_len=self.fp8_amax_history_len,
                amax_compute_algo=self.fp8_amax_compute_algo
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
        elif self.use_amp:
            with autocast(dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        grad_config = self.config.get("optimization", {}).get("gradients", {
            "clip_type": "norm",
            "clip_value": 1.0
        })
        if grad_config.get("clip_type") == "norm":
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                grad_config.get("clip_value", 1.0)
            )
        else:
            grad_norm = 0.0
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Calculate metrics
        step_time = time.time() - start_time
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        # Count actual tokens (exclude padding)
        actual_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item()
        tokens_processed = actual_tokens * self.world_size
        tokens_per_second = tokens_processed / step_time
        
        metrics = {
            "loss": loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "gradient_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "tokens_per_second": tokens_per_second,
            "step_time": step_time
        }
        
        return metrics
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=self.local_rank != 0
        )
        
        epoch_metrics = {
            "loss": [],
            "tokens_per_second": []
        }
        
        for step, batch in enumerate(progress_bar):
            # Update sequence length curriculum if applicable
            if hasattr(self.train_dataset, 'update_sequence_length'):
                current_step = epoch * len(dataloader) + step
                new_seq_len = self.train_dataset.update_sequence_length(current_step, self.total_steps)
                if self.local_rank == 0 and step % 100 == 0:
                    print(f"Sequence length updated to: {new_seq_len}")
            
            metrics = self.train_step(batch)
            
            # Update metrics
            epoch_metrics["loss"].append(metrics["loss"])
            epoch_metrics["tokens_per_second"].append(metrics["tokens_per_second"])
            
            # Update progress bar
            if self.local_rank == 0:
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{metrics['learning_rate']:.2e}",
                    "tokens/s": f"{metrics['tokens_per_second']:.0f}"
                })
                
                # Log to wandb
                if step % 100 == 0 and self.local_rank == 0:
                    try:
                        wandb.log(metrics, step=step)
                    except:
                        pass  # Wandb not initialized
        
        # Return epoch summary
        return {
            "avg_loss": sum(epoch_metrics["loss"]) / len(epoch_metrics["loss"]),
            "avg_tokens_per_second": sum(epoch_metrics["tokens_per_second"]) / len(epoch_metrics["tokens_per_second"])
        }
    
    def setup_data_pipeline(self, num_epochs: int):
        """Setup data loading pipeline."""
        train_config = self.config.get("training", {})
        data_config = self.config.get("data", train_config.get("data", {}))
        
        # Get dataset info
        dataset_name = data_config.get("dataset_name", "Yxanul/wikipedia-2k-high-quality")
        batch_size = int(train_config.get("batch_size", 8))
        max_length = int(data_config.get("max_sequence_length", 2048))
        
        # Estimate dataset size
        self.dataset_size = estimate_dataset_size(dataset_name)
        
        # Create dataloaders
        self.train_dataloader, self.train_dataset = create_dataloader(
            dataset_name=dataset_name,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            stage_config=train_config,
            num_workers=2,
            split="train"
        )
        
        # Store dataset reference for curriculum updates
        self.train_dataset = self.train_dataset
        
        # Create validation dataloader if configured
        validation_config = train_config.get("validation", {})
        if validation_config.get("validation_split", 0) > 0:
            print(f"Creating validation dataloader with {validation_config['validation_split']} split")
            self.val_dataloader, self.val_dataset = create_dataloader(
                dataset_name=dataset_name,
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                stage_config=train_config,
                num_workers=1,
                split=f"train[{int((1-validation_config['validation_split'])*100)}%:]"  # Last X% for validation
            )
        else:
            self.val_dataloader = None
            self.val_dataset = None
        
        # Calculate total training steps
        gradient_accumulation = int(train_config["training"].get("gradient_accumulation_steps", 1))
        self.total_steps = calculate_training_steps(
            dataset_size=self.dataset_size,
            batch_size=batch_size,
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation,
            world_size=self.world_size
        )
        
        print(f"Dataset: {dataset_name}")
        print(f"Estimated size: {self.dataset_size:,} examples")
        print(f"Total training steps: {self.total_steps:,}")
        
        # Recreate scheduler with correct steps
        self._setup_scheduler()
    
    def validate(self, dataloader=None, max_batches: int = 100):
        """Run validation and calculate perplexity."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        if dataloader is None:
            # Create validation dataloader if not provided
            if self.val_dataloader is None:
                return None
            dataloader = self.val_dataloader
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                    
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
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
                
                # Accumulate loss
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss))
            return perplexity.item()
        return None
    
    def save_checkpoint(self, epoch: int, step: int, path: str):
        """Save model checkpoint."""
        if self.local_rank == 0:
            checkpoint = {
                "epoch": epoch,
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
                "metrics": self.metrics
            }
            torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")
            
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "metrics" in checkpoint:
            self.metrics = checkpoint["metrics"]
        return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def setup_deepspeed(model, args, config_path: str):
    """Setup DeepSpeed if configured."""
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed not installed. Please install with: pip install deepspeed")
    
    # Load DeepSpeed config
    with open(config_path, 'r') as f:
        ds_config = yaml.safe_load(f)
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        config=ds_config
    )
    
    return model_engine, optimizer


def main():
    """Main training loop."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--stage", type=str, default="stage1_wikipedia")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--validate_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--deepspeed", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create model
    from model import create_model, ModelConfig
    
    # Load model config
    with open(f"{args.config_dir}/model_config.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
    
    model = create_model(model_config["model"])
    
    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        config_path=args.config_dir,
        stage=args.stage,
        local_rank=args.local_rank
    )
    
    # Setup data pipeline
    trainer.setup_data_pipeline(args.num_epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_epoch, start_step = trainer.load_checkpoint(args.resume_from)
    
    # Training loop
    print("Starting training...")
    print(f"Training on {trainer.world_size} GPUs")
    print(f"Starting from epoch {start_epoch}, step {start_step}")
    
    global_step = start_step
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Training
        epoch_metrics = trainer.train_epoch(trainer.train_dataloader, epoch + 1)
        
        # Validation
        if trainer.val_dataloader is not None:
            perplexity = trainer.validate()
            if perplexity:
                print(f"Validation Perplexity: {perplexity:.2f}")
                trainer.metrics["perplexity"].append(perplexity)
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/epoch_{epoch + 1}.pt"
        trainer.save_checkpoint(epoch + 1, global_step, checkpoint_path)
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {epoch_metrics['avg_loss']:.4f}")
        print(f"  Average Tokens/sec: {epoch_metrics['avg_tokens_per_second']:.0f}")
    
    print("\nTraining completed!")
    

if __name__ == "__main__":
    main()