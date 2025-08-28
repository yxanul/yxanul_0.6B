#!/usr/bin/env python3
"""
Test generation from a trained TinyStories model.
Load checkpoint and generate various samples to check quality.
"""

import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path
from model import ModelConfig, SimpleGPT

def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same config as training
    config = ModelConfig(
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=3,  # GQA with 4x compression
        block_size=128,
        dropout=0.0,
        bias=False
    )
    
    model = SimpleGPT(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    if 'iter_num' in checkpoint:
        print(f"Checkpoint from iteration: {checkpoint['iter_num']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model

@torch.no_grad()
def generate(model, prompt: str, max_tokens: int = 150, temperature: float = 0.8, top_k: int = 50, device: str = 'cuda'):
    """Generate text from prompt."""
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated = []
    model.eval()
    
    for _ in range(max_tokens):
        # Forward pass
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
            generated = generate(model, prompt, max_tokens=100, temperature=temp, top_k=40, device=device)
            print(f"Temp {temp}: {generated}\n")
    
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
        generated = generate(model, prompt, max_tokens=150, temperature=0.8, top_k=50, device=device)
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
            "checkpoints_tinystories/ckpt_1000.pt",
            "checkpoints_tinystories/ckpt_5000.pt",
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if Path(path).exists():
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            print("No checkpoint found! Please specify path:")
            print("  python test_generation.py <checkpoint_path>")
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