#!/usr/bin/env python3
"""
Test script for evaluating the trained FP8 model's generation capabilities.
This script loads the checkpoint and generates text with various prompts.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
import time
from typing import Optional, List
import argparse

# Import the appropriate model based on device
import sys

# Check if we should even try to import TE
TE_AVAILABLE = False
if torch.cuda.is_available():
    try:
        import transformer_engine
        TE_AVAILABLE = True
    except ImportError:
        pass

# Now import the appropriate model
if TE_AVAILABLE:
    from model_te_clean import ModelConfig, CleanGPT_TE
    print("Using TransformerEngine model (GPU)")
else:
    from model_inference import ModelConfig, GPTInference
    print("Using CPU inference model")

# Tokenizer will be imported in get_tokenizer function


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint - need weights_only=False for compatibility with training checkpoint
    # This is safe since we created this checkpoint ourselves
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract config and create model
    config_dict = checkpoint['config']
    
    # Create config based on available backend
    if TE_AVAILABLE and device == 'cuda':
        config = ModelConfig(
            vocab_size=config_dict['vocab_size'],
            n_layer=config_dict['n_layer'],
            n_head=config_dict['n_head'],
            n_embd=config_dict['n_embd'],
            n_kv_heads=config_dict['n_kv_heads'],
            block_size=config_dict['block_size'],
            dropout=0.0,  # No dropout for inference
            use_fp8=False,  # No FP8 for inference
        )
        model = CleanGPT_TE(config)
        model.load_state_dict(checkpoint['model'])
    else:
        # Use CPU inference model
        config = ModelConfig(
            vocab_size=config_dict['vocab_size'],
            n_layer=config_dict['n_layer'],
            n_head=config_dict['n_head'],
            n_embd=config_dict['n_embd'],
            n_kv_heads=config_dict['n_kv_heads'],
            block_size=config_dict['block_size'],
            dropout=0.0,  # No dropout for inference
        )
        model = GPTInference(config)
        # Load weights with mapping from TE format
        model.load_from_te_checkpoint(checkpoint_path)
    
    model.eval()
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        model = model.to(torch.bfloat16)
    else:
        device = 'cpu'
        model = model.to(torch.float32)
    
    print(f"Model loaded successfully!")
    print(f"  - Iterations trained: {checkpoint['iter_num']}")
    print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  - Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"  - Device: {device}")
    print()
    
    return model, config


def get_tokenizer(vocab_size: int):
    """Get the appropriate tokenizer based on vocab size."""
    # Import transformers for SmolLM tokenizer
    from transformers import AutoTokenizer
    
    if vocab_size == 49152:
        # SmolLM2-135M tokenizer - EXACT tokenizer used for training!
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/SmolLM2-135M",
                use_fast=True
            )
            print("Using SmolLM2-135M tokenizer (training tokenizer)")
        except:
            # Fallback to SmolLM v1 if v2 not available
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "HuggingFaceTB/SmolLM-1.7B",
                    use_fast=True
                )
                print("Using SmolLM-1.7B tokenizer (fallback)")
            except:
                print("ERROR: Could not load SmolLM tokenizer!")
                raise
        print(f"  Vocab size: {len(tokenizer)}")
        return tokenizer
    else:
        # For other vocab sizes, still try SmolLM
        print(f"Warning: Unexpected vocab_size={vocab_size}, using SmolLM tokenizer anyway")
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            use_fast=True
        )
        return tokenizer


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = 'cuda'
):
    """Generate text from a prompt."""
    # Encode prompt (handle both tiktoken and HF tokenizers)
    if hasattr(tokenizer, 'encode'):
        # HuggingFace tokenizer
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        # tiktoken tokenizer (fallback)
        tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    model.eval()
    generated_tokens = []
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    for _ in range(max_new_tokens):
        # Get logits
        with torch.no_grad():
            logits, _ = model(x)
        
        # Get last token logits
        logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        x = torch.cat([x, next_token], dim=1)
        generated_tokens.append(next_token.item())
        
        # Stop if we hit end token (if applicable)
        # Skip this for HF tokenizers as they handle differently
        if hasattr(tokenizer, 'eos_token_id'):
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        print(".", end="", flush=True)
    
    print(" Done!")
    
    # Decode
    generated_text = tokenizer.decode(generated_tokens)
    full_text = prompt + generated_text
    
    return full_text


def run_test_suite(model, tokenizer, device='cuda'):
    """Run a comprehensive test suite with various prompts."""
    
    test_prompts = [
        # Story generation
        ("Once upon a time, in a small village", 150, 0.8),
        
        # Technical writing
        ("The key principles of machine learning are", 100, 0.7),
        
        # Creative writing
        ("The robot looked at the stars and wondered", 120, 0.9),
        
        # Dialogue
        ("'Hello,' she said. 'I've been waiting for you.'", 100, 0.8),
        
        # Educational
        ("To bake a chocolate cake, you will need", 100, 0.7),
        
        # News style
        ("Breaking news: Scientists have discovered", 80, 0.7),
        
        # Code comments (testing technical understanding)
        ("def fibonacci(n):\n    # This function", 60, 0.6),
    ]
    
    print("="*60)
    print("COMPREHENSIVE MODEL TEST SUITE")
    print("="*60)
    
    for i, (prompt, max_tokens, temp) in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}] Temperature={temp}")
        print("-"*60)
        
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=50,
            top_p=0.9,
            device=device
        )
        
        print(f"Generated text:\n{generated}")
        print("-"*60)
        time.sleep(0.5)  # Brief pause between generations
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE!")
    print("="*60)


def interactive_mode(model, tokenizer, device='cuda'):
    """Interactive generation mode."""
    print("\n" + "="*60)
    print("INTERACTIVE GENERATION MODE")
    print("Type 'quit' to exit, 'help' for options")
    print("="*60)
    
    # Default settings
    settings = {
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
        'max_tokens': 100
    }
    
    while True:
        print(f"\nCurrent settings: temp={settings['temperature']}, "
              f"top_k={settings['top_k']}, top_p={settings['top_p']}, "
              f"max_tokens={settings['max_tokens']}")
        
        prompt = input("\nEnter prompt (or command): ").strip()
        
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'help':
            print("\nCommands:")
            print("  quit - Exit interactive mode")
            print("  help - Show this help")
            print("  temp=X - Set temperature (0.1-2.0)")
            print("  topk=X - Set top-k (1-100)")
            print("  topp=X - Set top-p (0.1-1.0)")
            print("  max=X - Set max tokens (10-500)")
            continue
        elif prompt.startswith('temp='):
            try:
                settings['temperature'] = float(prompt.split('=')[1])
                print(f"Temperature set to {settings['temperature']}")
            except:
                print("Invalid temperature value")
            continue
        elif prompt.startswith('topk='):
            try:
                settings['top_k'] = int(prompt.split('=')[1])
                print(f"Top-k set to {settings['top_k']}")
            except:
                print("Invalid top-k value")
            continue
        elif prompt.startswith('topp='):
            try:
                settings['top_p'] = float(prompt.split('=')[1])
                print(f"Top-p set to {settings['top_p']}")
            except:
                print("Invalid top-p value")
            continue
        elif prompt.startswith('max='):
            try:
                settings['max_tokens'] = int(prompt.split('=')[1])
                print(f"Max tokens set to {settings['max_tokens']}")
            except:
                print("Invalid max tokens value")
            continue
        
        if prompt:
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=settings['max_tokens'],
                temperature=settings['temperature'],
                top_k=settings['top_k'],
                top_p=settings['top_p'],
                device=device
            )
            print(f"\nGenerated:\n{generated}")


def main():
    parser = argparse.ArgumentParser(description='Test the trained model')
    parser.add_argument('checkpoint', type=str, nargs='?', 
                       default='best_model_fp8_optimized.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--suite', action='store_true',
                       help='Run full test suite')
    parser.add_argument('--prompt', type=str,
                       help='Single prompt to test')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint '{args.checkpoint}' not found!")
        print(f"Looking in: {Path(args.checkpoint).absolute()}")
        return
    
    # Load model
    model, config = load_model(args.checkpoint, args.device)
    
    # Get tokenizer
    tokenizer = get_tokenizer(config.vocab_size)
    
    # Run appropriate mode
    if args.interactive:
        interactive_mode(model, tokenizer, args.device)
    elif args.suite:
        run_test_suite(model, tokenizer, args.device)
    elif args.prompt:
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device
        )
        print(f"\nGenerated:\n{generated}")
    else:
        # Default: Run a few sample generations
        print("\nRunning sample generations...")
        print("="*60)
        
        samples = [
            "The future of artificial intelligence",
            "Once there was a little dragon who",
            "The most important thing to remember is",
        ]
        
        for prompt in samples:
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=80,
                temperature=0.8,
                device=args.device
            )
            print(f"\nGenerated:\n{generated}")
            print("-"*60)


if __name__ == "__main__":
    main()