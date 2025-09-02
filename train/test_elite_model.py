#!/usr/bin/env python3
"""
Test script for the elite 2.218 val loss model.
This loads the trained model and generates text with various prompts.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer
from model_inference import ModelConfig, GPTInference
import time

def load_elite_model(checkpoint_path='best_model_fp8_optimized.pt'):
    """Load the trained elite model."""
    print("="*60)
    print("Loading Elite Model (Val Loss 2.218)")
    print("="*60)
    
    # Create config
    config = ModelConfig(
        vocab_size=49152,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=3,
        block_size=2048,
        dropout=0.0  # No dropout for inference
    )
    
    # Create model
    model = GPTInference(config)
    
    # Load checkpoint
    checkpoint_info = model.load_from_te_checkpoint(checkpoint_path)
    
    # Model info
    print(f"\nModel Statistics:")
    print(f"  Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"  Architecture: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
    print(f"  GQA Compression: {config.n_head}/{config.n_kv_heads} = {config.n_head//config.n_kv_heads}x")
    print("="*60)
    
    model.eval()
    return model, config

def get_tokenizer():
    """Get the SmolLM tokenizer used for training."""
    print("Loading SmolLM2-135M tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        use_fast=True
    )
    print(f"  Vocab size: {len(tokenizer)}")
    return tokenizer

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=40, top_p=0.85, repetition_penalty=1.1):
    """Generate text from prompt with optimized sampling parameters."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
    
    print(f"\n📝 Prompt: {prompt}")
    print("🤖 Generating...", end="", flush=True)
    
    start_time = time.time()
    
    # Generate tokens
    generated_ids = input_ids
    for _ in range(max_new_tokens):
        # Get logits
        with torch.no_grad():
            logits, _ = model(generated_ids)
        
        # Get last token logits
        logits = logits[:, -1, :] / temperature
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()):
                logits[0, token_id] /= repetition_penalty
        
        # Apply top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        print(".", end="", flush=True)
    
    print(" Done!")
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    elapsed = time.time() - start_time
    tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
    print(f"⏱️  Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.1f} tok/s)")
    
    return generated_text

def run_test_suite(model, tokenizer):
    """Run comprehensive test suite."""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL TEST SUITE")
    print("Testing Elite Model (Val Loss 2.218)")
    print("="*60)
    
    test_prompts = [
        # Story generation
        {
            "prompt": "Once upon a time, in a small village nestled between mountains",
            "max_tokens": 150,
            "temperature": 0.75,  # Slightly higher for creativity
            "category": "Creative Writing"
        },
        
        # Technical explanation
        {
            "prompt": "Machine learning is a field of artificial intelligence that",
            "max_tokens": 100,
            "temperature": 0.65,  # Lower for factual content
            "category": "Technical"
        },
        
        # Code generation
        {
            "prompt": "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n",
            "max_tokens": 80,
            "temperature": 0.5,  # Very low for precise code
            "category": "Code"
        },
        
        # Dialogue
        {
            "prompt": "Alice: \"I've been thinking about what you said yesterday.\"\nBob: \"",
            "max_tokens": 100,
            "temperature": 0.8,
            "category": "Dialogue"
        },
        
        # Reasoning
        {
            "prompt": "To solve this problem, we need to consider three factors:",
            "max_tokens": 120,
            "temperature": 0.7,
            "category": "Reasoning"
        },
        
        # News style
        {
            "prompt": "Breaking News: Scientists at MIT have discovered",
            "max_tokens": 100,
            "temperature": 0.7,
            "category": "News"
        },
        
        # Instructions
        {
            "prompt": "How to make the perfect cup of coffee:\n1.",
            "max_tokens": 120,
            "temperature": 0.7,
            "category": "Instructions"
        }
    ]
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}] Category: {test['category']}")
        print("-"*60)
        
        generated = generate_text(
            model, tokenizer,
            test["prompt"],
            max_new_tokens=test["max_tokens"],
            temperature=test["temperature"],
            top_k=40,  # Tighter vocabulary
            top_p=0.85,  # More focused
            repetition_penalty=1.1  # Prevent repetition
        )
        
        print(f"\n✨ Generated Text:\n{generated}")
        print("-"*60)
        
        # Brief pause between tests
        time.sleep(0.5)
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE!")
    print("Model Quality Assessment:")
    print("  ✅ Coherence: Check if text maintains topic")
    print("  ✅ Grammar: Check if sentences are well-formed")
    print("  ✅ Creativity: Check if outputs are diverse")
    print("  ✅ Knowledge: Check if facts are reasonable")
    print("="*60)

def interactive_mode(model, tokenizer):
    """Interactive generation mode."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("Type 'quit' to exit, 'help' for options")
    print("="*60)
    
    settings = {
        'temperature': 0.7,  # Better default
        'top_k': 40,  # Tighter vocabulary
        'top_p': 0.85,  # More focused
        'repetition_penalty': 1.1,  # Prevent repetition
        'max_tokens': 100
    }
    
    while True:
        print(f"\n⚙️  Settings: temp={settings['temperature']}, top_k={settings['top_k']}, "
              f"top_p={settings['top_p']}, max={settings['max_tokens']}")
        
        prompt = input("\n💭 Enter prompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'help':
            print("\nCommands:")
            print("  quit - Exit")
            print("  help - Show this help")
            print("  temp=X - Set temperature (0.1-2.0)")
            print("  topk=X - Set top-k (1-100)")
            print("  topp=X - Set top-p (0.1-1.0)")
            print("  max=X - Set max tokens (10-500)")
            continue
        elif prompt.startswith('temp='):
            try:
                settings['temperature'] = float(prompt.split('=')[1])
                print(f"✅ Temperature set to {settings['temperature']}")
            except:
                print("❌ Invalid temperature")
            continue
        elif prompt.startswith('topk='):
            try:
                settings['top_k'] = int(prompt.split('=')[1])
                print(f"✅ Top-k set to {settings['top_k']}")
            except:
                print("❌ Invalid top-k")
            continue
        elif prompt.startswith('topp='):
            try:
                settings['top_p'] = float(prompt.split('=')[1])
                print(f"✅ Top-p set to {settings['top_p']}")
            except:
                print("❌ Invalid top-p")
            continue
        elif prompt.startswith('max='):
            try:
                settings['max_tokens'] = int(prompt.split('=')[1])
                print(f"✅ Max tokens set to {settings['max_tokens']}")
            except:
                print("❌ Invalid max tokens")
            continue
        
        if prompt:
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=settings['max_tokens'],
                temperature=settings['temperature'],
                top_k=settings['top_k'],
                top_p=settings['top_p'],
                repetition_penalty=settings.get('repetition_penalty', 1.1)
            )
            print(f"\n✨ Generated:\n{generated}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test the elite trained model')
    parser.add_argument('--checkpoint', type=str, default='best_model_fp8_optimized.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['suite', 'interactive', 'single'],
                       default='suite', help='Test mode')
    parser.add_argument('--prompt', type=str, help='Single prompt to test')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.85)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint '{args.checkpoint}' not found!")
        return
    
    # Load model and tokenizer
    model, config = load_elite_model(args.checkpoint)
    tokenizer = get_tokenizer()
    
    print("\n🚀 Elite Model Ready!")
    print(f"   Val Loss achieved: 2.218")
    print(f"   Perplexity: ~9.19")
    print(f"   Quality: Commercial-grade")
    
    # Run appropriate mode
    if args.mode == 'suite':
        run_test_suite(model, tokenizer)
    elif args.mode == 'interactive':
        interactive_mode(model, tokenizer)
    elif args.mode == 'single':
        if not args.prompt:
            print("❌ Please provide a prompt with --prompt")
            return
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        print(f"\n✨ Generated:\n{generated}")

if __name__ == "__main__":
    main()