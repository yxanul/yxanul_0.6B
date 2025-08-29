#!/usr/bin/env python3
"""
Test generation from a trained textbook model with SuperBPE tokenizer.
Handles both GPT-2 and SuperBPE vocabularies.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from model import ModelConfig, SimpleGPT
from transformers import AutoTokenizer
import platform

# Dummy TrainingConfig to allow unpickling
@dataclass
class TrainingConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 200005  # Updated for SuperBPE
    block_size: int = 128
    dropout: float = 0.05
    use_factorized_embedding: bool = True
    embedding_rank: int = 128
    batch_size: int = 32
    gradient_accumulation_steps: int = 32
    max_iters: int = 1000
    eval_interval: int = 200
    eval_iters: int = 100
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    warmup_iters: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = False
    log_interval: int = 100
    checkpoint_interval: int = 5000
    checkpoint_dir: str = 'checkpoints_tinystories'
    wandb_project: str = 'tinystories-precision-test'
    wandb_run_name: Optional[str] = None
    data_dir: str = 'data_textbooks_superbpe'
    use_superbpe: bool = True

def load_tokenizer(vocab_size: int):
    """Load appropriate tokenizer based on vocabulary size."""
    if vocab_size > 100000:  # SuperBPE
        print("Loading SuperBPE tokenizer...")
        
        # Determine cache path based on platform
        if platform.system() == "Linux":
            cache_base = Path("/workspace/yxanul_0.6B/tokenizer_cache")
        else:
            cache_base = Path("D:/ai_testing/yxanul_0.6B/tokenizer_cache")
        
        if not cache_base.exists():
            cache_base = Path("../tokenizer_cache")
        
        t80k_path = cache_base / "superbpe-t80k-fast"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(t80k_path),
                use_fast=True,
                local_files_only=True
            )
            print(f"Loaded SuperBPE tokenizer (vocab size: {len(tokenizer)})")
            return tokenizer
        except:
            print("Warning: Could not load SuperBPE, falling back to GPT-2")
    
    # Fallback to GPT-2
    print("Loading GPT-2 tokenizer...")
    import tiktoken
    return tiktoken.get_encoding("gpt2")

def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint
    if 'config' in checkpoint:
        training_config = checkpoint['config']
        vocab_size = getattr(training_config, 'vocab_size', 200005)
        
        config = ModelConfig(
            vocab_size=vocab_size,
            n_layer=getattr(training_config, 'n_layer', 12),
            n_head=getattr(training_config, 'n_head', 12),
            n_embd=getattr(training_config, 'n_embd', 768),
            n_kv_heads=getattr(training_config, 'n_head', 12) // 4,
            block_size=getattr(training_config, 'block_size', 128),
            dropout=0.0,  # No dropout for inference
            bias=False,
            use_factorized_embedding=getattr(training_config, 'use_factorized_embedding', True),
            embedding_rank=getattr(training_config, 'embedding_rank', 128)
        )
        print(f"Loaded config: vocab_size={vocab_size}, factorized={config.use_factorized_embedding}")
    else:
        # Default to SuperBPE config
        vocab_size = 200005
        config = ModelConfig(
            vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768,
            n_kv_heads=3,
            block_size=128,
            dropout=0.0,
            bias=False,
            use_factorized_embedding=True,
            embedding_rank=128
        )
        print("Using default SuperBPE config")
    
    model = SimpleGPT(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    if 'iter_num' in checkpoint:
        print(f"Checkpoint from iteration: {checkpoint['iter_num']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        print(f"Perplexity: {torch.exp(torch.tensor(checkpoint['best_val_loss'])).item():.2f}")
    
    return model, vocab_size

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 150, temperature: float = 0.8, top_k: int = 50, device: str = 'cuda'):
    """Generate text from prompt with sliding window for long sequences."""
    
    # Get block size from model config
    block_size = model.config.block_size
    
    # Handle different tokenizer types
    if hasattr(tokenizer, 'encode'):
        if hasattr(tokenizer, 'encode_ordinary'):  # tiktoken
            tokens = tokenizer.encode_ordinary(prompt)
            eos_token = tokenizer.eot_token
        else:  # transformers tokenizer
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            eos_token = tokenizer.eos_token_id if tokenizer.eos_token_id else 200004
    else:
        raise ValueError("Unknown tokenizer type")
    
    # Truncate prompt if too long
    if len(tokens) >= block_size:
        print(f"Warning: Prompt too long ({len(tokens)} tokens), truncating to {block_size-1}")
        tokens = tokens[-(block_size-1):]
    
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated = []
    model.eval()
    
    for _ in range(max_tokens):
        # Use sliding window if sequence exceeds block size
        if x.size(1) >= block_size:
            x_input = x[:, -block_size:]
        else:
            x_input = x
        
        # Forward pass
        logits, _ = model(x_input)
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
        if idx_next.item() == eos_token:
            break
    
    # Decode full sequence
    full_tokens = tokens + generated
    
    if hasattr(tokenizer, 'decode'):
        return tokenizer.decode(full_tokens)
    else:  # tiktoken
        return tokenizer.decode(full_tokens)

def test_model(checkpoint_path: str):
    """Test model with educational prompts."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model and get vocab size
    model, vocab_size = load_checkpoint(checkpoint_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.1f}M\n")
    
    # Load appropriate tokenizer
    tokenizer = load_tokenizer(vocab_size)
    
    # Educational test prompts (not stories!)
    test_prompts = [
        "A library catalog is",
        "The main purpose of",
        "To find information about",
        "When learning a new skill, you should",
        "The difference between",
        "In mathematics, the concept of",
        "Scientific research involves",
        "The process of writing involves",
    ]
    
    print("=" * 60)
    print("GENERATION TESTS - EDUCATIONAL CONTENT")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] Prompt: {prompt}")
        print("-" * 40)
        
        # Generate with different temperatures
        for temp in [0.7, 0.9]:
            generated = generate(model, tokenizer, prompt, max_tokens=100, temperature=temp, top_k=40, device=device)
            print(f"Temp {temp}: {generated}\n")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)
    print("Try educational prompts like:")
    print("  - 'The steps to solve a problem are'")
    print("  - 'A database is used for'")
    print("  - 'Learning requires'")
    
    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt.strip():
            print("Please enter a prompt!")
            continue
        
        print("\nGenerating...")
        generated = generate(model, tokenizer, prompt, max_tokens=150, temperature=0.8, top_k=50, device=device)
        print(f"\nGenerated: {generated}")

def main():
    import sys
    
    # Check for checkpoint path argument
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Default paths to check
        possible_paths = [
            "checkpoints_tinystories/best_model.pt",
            "checkpoints_textbooks/best_model.pt",
            "checkpoints_superbpe/best_model.pt",
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if Path(path).exists():
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            print("No checkpoint found! Please specify path:")
            print("  python test_textbook_generation.py <checkpoint_path>")
            print("\nOr ensure one of these exists:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"Testing checkpoint: {checkpoint_path}")
    test_model(checkpoint_path)

if __name__ == "__main__":
    main()