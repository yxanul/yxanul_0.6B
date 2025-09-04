"""
Comprehensive test to verify MoE model alignment and readiness for training.
Tests model instantiation, forward pass, loss computation, and configuration consistency.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import traceback

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(msg, status='info'):
    if status == 'success':
        print(f"{GREEN}✓ {msg}{RESET}")
    elif status == 'error':
        print(f"{RED}✗ {msg}{RESET}")
    elif status == 'warning':
        print(f"{YELLOW}⚠ {msg}{RESET}")
    else:
        print(f"{BLUE}ℹ {msg}{RESET}")

def test_model_import():
    """Test if model can be imported."""
    print_status("Testing model import...", 'info')
    try:
        from model_moe import (
            ModelConfig,
            OptimizedGPT_GLMini_PRMoE,
            build_glm_mini_prmoe_mtp,
            PyramidResidualMoE
        )
        print_status("Model modules imported successfully", 'success')
        return True
    except ImportError as e:
        print_status(f"Failed to import model: {e}", 'error')
        print_status("Make sure transformer-engine is installed: pip install transformer-engine", 'warning')
        return False
    except Exception as e:
        print_status(f"Unexpected error importing model: {e}", 'error')
        return False

def test_model_instantiation():
    """Test model instantiation with default config."""
    print_status("\nTesting model instantiation...", 'info')
    try:
        from model_moe import build_glm_mini_prmoe_mtp
        
        model, cfg = build_glm_mini_prmoe_mtp()
        
        # Check configuration values
        assert cfg.vocab_size == 32000, f"Expected vocab_size=32000, got {cfg.vocab_size}"
        assert cfg.block_size == 2048, f"Expected block_size=2048, got {cfg.block_size}"
        assert cfg.n_head == 28, f"Expected n_head=28, got {cfg.n_head}"
        assert cfg.n_kv_heads == 7, f"Expected n_kv_heads=7, got {cfg.n_kv_heads}"
        assert cfg.n_head % cfg.n_kv_heads == 0, "n_head must be divisible by n_kv_heads for GQA"
        
        print_status(f"Model created with {model.num_parameters()/1e6:.2f}M parameters", 'success')
        print_status(f"Configuration: vocab={cfg.vocab_size}, layers={cfg.n_layer}, dim={cfg.n_embd}", 'success')
        print_status(f"GQA: {cfg.n_head} Q-heads, {cfg.n_kv_heads} KV-heads (4x repeat)", 'success')
        
        return model, cfg
    except Exception as e:
        print_status(f"Failed to instantiate model: {e}", 'error')
        traceback.print_exc()
        return None, None

def test_forward_pass(model, cfg):
    """Test forward pass with mock data."""
    print_status("\nTesting forward pass...", 'info')
    
    if model is None:
        print_status("Skipping forward pass test (no model)", 'warning')
        return False
    
    try:
        # Create mock input
        batch_size = 2
        seq_len = 64  # Small sequence for testing
        
        # Random token IDs
        idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        
        print_status(f"Input shape: {idx.shape}", 'info')
        
        # Move to CUDA if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        idx = idx.to(device)
        targets = targets.to(device)
        
        print_status(f"Using device: {device}", 'info')
        
        # Test without FP8 first (simpler)
        with torch.no_grad():
            logits, loss = model(idx, targets)
        
        # Check outputs
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size), f"Wrong logits shape: {logits.shape}"
        assert loss is not None, "Loss is None"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        
        print_status(f"Forward pass successful! Loss: {loss.item():.4f}", 'success')
        print_status(f"Logits shape: {logits.shape}", 'success')
        
        # Test MoE routing stats if available
        moe_layers = [m for m in model.modules() if isinstance(m, PyramidResidualMoE)]
        if moe_layers:
            print_status(f"Found {len(moe_layers)} MoE layers", 'info')
        
        return True
        
    except Exception as e:
        print_status(f"Forward pass failed: {e}", 'error')
        traceback.print_exc()
        return False

def test_training_config_alignment():
    """Test if training config aligns with model."""
    print_status("\nTesting training configuration alignment...", 'info')
    
    try:
        from train_moe import TrainingConfig
        from model_moe import ModelConfig
        
        tcfg = TrainingConfig()
        mcfg = ModelConfig()
        
        # Check critical alignments
        issues = []
        
        if tcfg.block_size != mcfg.block_size:
            issues.append(f"block_size mismatch: train={tcfg.block_size}, model={mcfg.block_size}")
        
        if tcfg.data_dir == "data_mixed_3b" and not Path(tcfg.data_dir).exists():
            print_status(f"Data directory '{tcfg.data_dir}' not found (expected, will create later)", 'warning')
        
        # Check training hyperparameters
        print_status(f"Training config:", 'info')
        print_status(f"  - Batch size: {tcfg.batch_size}", 'info')
        print_status(f"  - Grad accum: {tcfg.grad_accum_steps}", 'info')
        print_status(f"  - Effective batch: {tcfg.batch_size * tcfg.grad_accum_steps}", 'info')
        print_status(f"  - Learning rate: {tcfg.learning_rate} (changed from 3e-4 to 6e-4)", 'info')
        print_status(f"  - Block size: {tcfg.block_size}", 'info')
        print_status(f"  - Max iters: {tcfg.max_iters}", 'info')
        
        if issues:
            for issue in issues:
                print_status(issue, 'error')
            return False
        else:
            print_status("Training and model configs align properly", 'success')
            return True
            
    except Exception as e:
        print_status(f"Failed to check training config: {e}", 'error')
        return False

def test_memory_requirements():
    """Estimate memory requirements."""
    print_status("\nEstimating memory requirements...", 'info')
    
    try:
        from model_moe import build_glm_mini_prmoe_mtp
        model, cfg = build_glm_mini_prmoe_mtp()
        
        n_params = model.num_parameters()
        
        # Memory estimates
        model_bf16 = n_params * 2 / 1e9  # BF16
        model_fp8 = n_params * 1 / 1e9   # FP8
        optimizer = n_params * 8 / 1e9   # AdamW states (FP32 momentum + variance)
        
        # Activation memory (rough estimate for batch_size=8, seq_len=2048)
        batch_size = 8
        seq_len = 2048
        n_layers = cfg.n_layer
        hidden = cfg.n_embd
        
        # Per layer: attention (Q,K,V,output) + FFN
        act_per_layer = batch_size * seq_len * hidden * 8 * 2 / 1e9  # BF16
        total_act = act_per_layer * n_layers
        
        print_status(f"Model parameters: {n_params/1e6:.2f}M", 'info')
        print_status(f"Memory requirements:", 'info')
        print_status(f"  - Model (BF16): {model_bf16:.2f} GB", 'info')
        print_status(f"  - Model (FP8): {model_fp8:.2f} GB", 'info')
        print_status(f"  - Optimizer: {optimizer:.2f} GB", 'info')
        print_status(f"  - Activations (batch=8): ~{total_act:.2f} GB", 'info')
        print_status(f"  - Total (BF16): ~{model_bf16 + optimizer + total_act:.2f} GB", 'warning')
        print_status(f"  - Total (FP8): ~{model_fp8 + optimizer + total_act:.2f} GB", 'success')
        
        return True
        
    except Exception as e:
        print_status(f"Failed to estimate memory: {e}", 'error')
        return False

def main():
    print("=" * 60)
    print(f"{BLUE}MoE Model Alignment Test{RESET}")
    print("=" * 60)
    
    # Track results
    all_passed = True
    
    # Run tests
    if not test_model_import():
        print_status("\nCannot proceed without model import", 'error')
        print_status("Install transformer-engine: pip install transformer-engine", 'warning')
        return False
    
    model, cfg = test_model_instantiation()
    if model is None:
        all_passed = False
    
    # Import PyramidResidualMoE for forward pass test
    from model_moe import PyramidResidualMoE
    
    if not test_forward_pass(model, cfg):
        all_passed = False
    
    if not test_training_config_alignment():
        all_passed = False
    
    if not test_memory_requirements():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print_status("All tests passed! Model is ready for training.", 'success')
        print_status("\nNext steps:", 'info')
        print_status("1. Create your tokenized data (train.bin, val.bin)", 'info')
        print_status("2. Place them in 'data_mixed_3b' directory", 'info')
        print_status("3. Install transformer-engine if not already done", 'info')
        print_status("4. Run: python train_moe.py", 'info')
    else:
        print_status("Some tests failed. Please fix issues before training.", 'error')
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)