#!/usr/bin/env python3
"""
Training script using the EnhancedTrainer with comprehensive monitoring.
This shows how to use the enhanced monitoring in practice.
"""

import os
import sys
import torch
import yaml
from dataclasses import fields

sys.path.append('src')

from model import create_model, ModelConfig
from enhanced_trainer import EnhancedTrainer

def main():
    """Main training function with enhanced monitoring."""
    
    print("=" * 60)
    print("Yxanul 177M Training with Enhanced Monitoring")
    print("=" * 60)
    
    # Load model configuration
    with open("configs/model_config.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Filter config for model creation
    valid_fields = {f.name for f in fields(ModelConfig)}
    filtered_config = {k: v for k, v in model_config["model"].items() if k in valid_fields}
    
    # Create model
    print("\n1. Creating Model...")
    model = create_model(filtered_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params/1e6:.1f}M")
    print(f"   Trainable parameters: {trainable_params/1e6:.1f}M")
    print(f"   Model architecture:")
    print(f"     - Layers: {filtered_config['num_layers']}")
    print(f"     - Hidden size: {filtered_config['hidden_size']}")
    print(f"     - Attention heads: {filtered_config['num_attention_heads']}")
    print(f"     - KV heads: {filtered_config['num_kv_heads']} (GQA)")
    print(f"     - FFN size: {filtered_config['intermediate_size']} (SwiGLU optimized)")
    print(f"     - Factorized embeddings: r={filtered_config.get('factorization_dim', 128)}")
    
    # Initialize trainer with enhanced monitoring
    print("\n2. Initializing Enhanced Trainer...")
    
    # In real usage with distributed training:
    # trainer = EnhancedTrainer(
    #     model=model,
    #     config_path="configs/stage1_wikipedia.yaml",
    #     stage="stage1",
    #     local_rank=int(os.environ.get("LOCAL_RANK", -1))
    # )
    
    print("   Enhanced monitoring features:")
    print("     [OK] 40+ metrics per training step")
    print("     [OK] Gradient health monitoring (explosion/vanishing/NaN)")
    print("     [OK] Memory leak detection")
    print("     [OK] Performance profiling (forward/backward/optimizer)")
    print("     [OK] Weight statistics (dead neurons, distributions)")
    print("     [OK] Activation statistics (layer outputs)")
    print("     [OK] Comprehensive validation metrics")
    print("     [OK] WandB integration with custom charts")
    
    # Training configuration
    print("\n3. Training Configuration:")
    print("   Stage 1: Wikipedia Foundation")
    print("     - Dataset: Yxanul/wikipedia-2k-high-quality")
    print("     - Total tokens: ~1B")
    print("     - Batch size: 512 (dynamic with curriculum)")
    print("     - Learning rate: 2e-4 with cosine schedule")
    print("     - Mixed precision: BF16")
    print("     - Gradient clipping: 1.0")
    
    # Monitoring dashboard
    print("\n4. WandB Dashboard Setup:")
    print("   The enhanced trainer will create a WandB dashboard with:")
    print("\n   Real-time Charts:")
    print("     - Loss & Perplexity curves")
    print("     - Learning rate schedule")
    print("     - Gradient norm evolution")
    print("     - Memory usage over time")
    print("     - Tokens/second throughput")
    print("\n   Health Indicators:")
    print("     - Gradient explosion warnings (norm > 100)")
    print("     - Gradient vanishing alerts (norm < 1e-6)")
    print("     - NaN/Inf detection")
    print("     - Dead neuron tracking")
    print("     - Memory leak detection")
    print("\n   Performance Analysis:")
    print("     - Forward vs Backward time")
    print("     - Optimizer step duration")
    print("     - Data loading bottlenecks")
    print("     - GPU utilization")
    
    # Training loop structure
    print("\n5. Training Loop Structure:")
    print("""
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Enhanced train_step with 40+ metrics
            metrics = trainer.train_step(batch)
            
            # Automatic logging to WandB
            wandb.log(metrics)
            
            # Periodic validation
            if step % val_interval == 0:
                val_metrics = trainer.validate()
                wandb.log(val_metrics)
            
            # Checkpoint with metadata
            if step % checkpoint_interval == 0:
                trainer.save_checkpoint(f"checkpoint_{step}.pt")
        
        # Epoch summary
        trainer.log_epoch_summary(epoch, epoch_metrics)
    """)
    
    # Key metrics to watch
    print("\n6. Key Metrics to Monitor:")
    print("\n   Training Quality:")
    print("     - train/loss: Should decrease steadily")
    print("     - train/perplexity: Target < 10 for good model")
    print("     - val/perplexity: Should track training closely")
    print("     - epoch/train_val_gap: < 1.0 indicates good generalization")
    print("\n   Training Health:")
    print("     - optim/gradient_norm: Should be stable around 0.5-2.0")
    print("     - optim/num_nans: Must be 0")
    print("     - weights/*/dead_ratio: Should be < 0.1")
    print("\n   Performance:")
    print("     - perf/tokens_per_second: Target > 100k for 8x A100")
    print("     - memory/allocated_gb: Should plateau, not increase")
    print("     - perf/forward_time vs backward_time: Should be ~1:2 ratio")
    
    # Example metric values
    print("\n7. Expected Metric Ranges:")
    print("   After 1000 steps:")
    print("     - Loss: 3.5-4.5")
    print("     - Perplexity: 30-90")
    print("     - Accuracy: 0.3-0.4")
    print("     - Gradient norm: 0.5-2.0")
    print("     - Tokens/second: 100k-150k")
    print("\n   After 10000 steps:")
    print("     - Loss: 2.5-3.5")
    print("     - Perplexity: 12-30")
    print("     - Accuracy: 0.4-0.5")
    print("     - Learning rate: Decreasing per schedule")
    print("\n   After full training:")
    print("     - Loss: < 2.5")
    print("     - Perplexity: < 12")
    print("     - Accuracy: > 0.5")
    
    print("\n" + "=" * 60)
    print("Ready to Train!")
    print("=" * 60)
    print("\nTo start training with enhanced monitoring:")
    print("\n  Single GPU:")
    print("  python train_with_monitoring.py")
    print("\n  Multi-GPU (8x A100):")
    print("  torchrun --nproc_per_node=8 train_with_monitoring.py")
    print("\n  With DeepSpeed:")
    print("  deepspeed --num_gpus=8 train_with_monitoring.py \\")
    print("    --deepspeed_config configs/deepspeed_config.json")
    print("\nMonitor progress at: https://wandb.ai/your-project/yxanul-177m")
    print("=" * 60)

if __name__ == "__main__":
    main()