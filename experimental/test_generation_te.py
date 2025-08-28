#!/usr/bin/env python3
"""
Test generation from TE models (FP8 or BF16 trained).
Handles the specific configuration used in train_te_fair.py.
"""

import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Import TE model
from model_te import TEModelConfig, TETransformerGPT

# Dummy TrainingConfig for checkpoint compatibility
@dataclass
class TrainingConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50256  # FP8-aligned vocab
    block_size: int = 128
    dropout: float = 0.0
    batch_size: int = 64
    gradient_accumulation_steps: int = 16
    max_iters: int = 1000
    eval_interval: int = 200
    eval_iters: int = 100
    learning_rate: float = 1e-4
    min_lr: float = 5e-5
    warmup_iters: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    fp8_format: str = "HYBRID"
    fp8_margin: float = 0
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    fp8_calibration_steps: int = 10
    force_bf16: bool = False
    device: str = 'cuda'
    compile: bool = False
    log_interval: int = 100
    checkpoint_interval: int = 5000
    checkpoint_dir: str = 'checkpoints_te_fair'
    wandb_project: str = 'tinystories-precision-test'
    wandb_run_name: Optional[str] = None

def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load TE model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with same config as training
    config = TEModelConfig(
        vocab_size=50256,  # FP8-aligned
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=12,
        block_size=128,
        dropout=0.0,
        bias=False
    )
    # Set FFN size explicitly
    config.ffn_hidden_size = 2048
    
    model = TETransformerGPT(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    if 'iter_num' in checkpoint:
        print(f"Checkpoint from iteration: {checkpoint['iter_num']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    if 'precision' in checkpoint:
        print(f"Training precision: {checkpoint['precision']}")
    
    return model

@torch.no_grad()
def generate(model, prompt: str, max_tokens: int = 150, temperature: float = 0.8, 
             top_k: int = 50, device: str = 'cuda'):
    """Generate text from prompt using TE model."""
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated = []
    model.eval()
    
    # Check for TransformerEngine
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, DelayedScaling
        HAS_TE = True
        
        # Create FP8 recipe for inference
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=1,
            amax_compute_algo="max",
        )
    except ImportError:
        HAS_TE = False
        fp8_recipe = None
    
    for _ in range(max_tokens):
        # Crop to block size if needed
        if x.size(1) > 128:  # block_size
            x = x[:, -128:]
        
        # Forward pass (with optional FP8 for inference)
        if HAS_TE:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, _ = model(x)
        else:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(x)
        
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        x = torch.cat((x, idx_next), dim=1)
        generated.append(idx_next.item())
        
        # Stop at EOS
        if idx_next.item() == enc.eot_token:
            break
    
    # Decode full sequence
    full_tokens = tokens + generated
    return enc.decode(full_tokens)

def test_model(checkpoint_path: str):
    """Test model with various prompts."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_checkpoint(checkpoint_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.1f}M\n")
    
    # Test prompts - typical TinyStories beginnings
    test_prompts = [
        "Once upon a time there was a little",
        "The little girl was playing in the",
        "Tom and his dog went to the",
        "One sunny day, a cat named",
        "In a small house lived a",
        "The rabbit hopped through the",
        "A brave knight rode his",
        "Sally found a magic",
    ]
    
    print("=" * 60)
    print("GENERATION TESTS")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] Prompt: {prompt}")
        print("-" * 40)
        
        # Generate with different temperatures
        for temp in [0.7, 0.9]:
            try:
                generated = generate(model, prompt, max_tokens=100, 
                                   temperature=temp, top_k=40, device=device)
                print(f"Temp {temp}: {generated}\n")
            except Exception as e:
                print(f"Error generating with temp {temp}: {e}\n")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt.strip():
            print("Please enter a prompt!")
            continue
        
        print("\nGenerating...")
        try:
            generated = generate(model, prompt, max_tokens=150, 
                               temperature=0.8, top_k=50, device=device)
            print(f"\nGenerated: {generated}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    import sys
    
    # Check for checkpoint path argument
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Default paths to check
        possible_paths = [
            "checkpoints_te_fair/best_model_fp8-hybrid.pt",
            "checkpoints_te_fair/best_model_bf16.pt",
            "checkpoints_te_hybrid/best_model.pt",
            "checkpoints_te/best_model.pt",
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if Path(path).exists():
                checkpoint_path = path
                print(f"Found checkpoint: {path}")
                break
        
        if checkpoint_path is None:
            print("No checkpoint found! Please specify path:")
            print("  python test_generation_te.py <checkpoint_path>")
            print("\nOr ensure one of these exists:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"Testing checkpoint: {checkpoint_path}")
    print("=" * 60)
    test_model(checkpoint_path)

if __name__ == "__main__":
    main()