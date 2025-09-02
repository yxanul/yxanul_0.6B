#!/usr/bin/env python3
"""
CPU-optimized test script for SFT model with low temperature for quality testing.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer
from model_inference import ModelConfig, GPTInference
import time
import warnings
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

def load_sft_model_cpu(checkpoint_path='best_sft_model.pt'):
    """Load the SFT model for CPU inference."""
    print("="*60)
    print("Loading SFT Model (CPU Mode)")
    print("="*60)
    
    # Create config
    config = ModelConfig(
        vocab_size=49152,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=3,
        block_size=2048,
        dropout=0.0
    )
    
    # Create model (stays on CPU)
    model = GPTInference(config)
    print("  Running on: CPU (will be slower)")
    
    # Load checkpoint
    checkpoint_info = model.load_from_te_checkpoint(checkpoint_path)
    
    print(f"\nModel Statistics:")
    print(f"  Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"  Temperature: 0.1 (deterministic mode)")
    print("="*60)
    
    model.eval()
    return model, config

def get_tokenizer():
    """Get the SmolLM tokenizer."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        use_fast=True
    )
    return tokenizer

@torch.no_grad()
def generate_response_cpu(model, tokenizer, user_input, max_new_tokens=200, temperature=0.1, repetition_penalty=1.2):
    """Generate response with CPU-optimized settings and repetition penalty."""
    # Format as instruction
    prompt = f"User: {user_input}\nAssistant:"
    
    # Encode prompt (stays on CPU)
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
    
    print(f"\n[User]: {user_input}")
    print("[Assistant]: ", end="", flush=True)
    
    start_time = time.time()
    
    # Generate tokens
    generated_ids = input_ids
    response_start = len(generated_ids[0])
    
    for i in range(max_new_tokens):
        # Get logits
        with torch.no_grad():
            logits, _ = model(generated_ids)
        
        # Get last token logits with low temperature for deterministic output
        logits = logits[:, -1, :] / temperature
        
        # Apply repetition penalty to reduce repetitive text
        if repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()[response_start:]):  # Only penalize response tokens
                logits[0, token_id] /= repetition_penalty
        
        # Simple greedy/sampling for CPU
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Check for stopping
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
        if "User:" in tokenizer.decode(generated_ids[0][response_start:], skip_special_tokens=False) + token_text:
            break
        
        # Append
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # Progress indicator
        if i % 5 == 0:
            print(".", end="", flush=True)
    
    print(" Done!")
    
    # Decode and extract response
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    if "Assistant:" in full_text:
        response = full_text.split("Assistant:")[-1].strip()
        if "User:" in response:
            response = response.split("User:")[0].strip()
    else:
        response = full_text[len(f"User: {user_input}"):].strip()
    
    elapsed = time.time() - start_time
    tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
    print(f"[Time]: {tokens_generated} tokens in {elapsed:.1f}s ({tokens_generated/elapsed:.1f} tok/s)")
    
    return response

def run_quality_test_suite(model, tokenizer):
    """Run focused test suite with low temperature."""
    print("\n" + "="*60)
    print("QUALITY TEST SUITE (Temperature 0.1)")
    print("Testing with deterministic generation for consistency")
    print("="*60)
    
    test_cases = [
        {
            "input": "What is 2 + 2?",
            "category": "Simple Math",
            "expected": "Should answer 4"
        },
        {
            "input": "Complete this sentence: The sky is",
            "category": "Completion",
            "expected": "Should complete with 'blue' or similar"
        },
        {
            "input": "Name the capital of France.",
            "category": "Factual",
            "expected": "Should answer Paris"
        },
        {
            "input": "Is water wet? Answer with yes or no.",
            "category": "Binary Question",
            "expected": "Should give yes/no answer"
        },
        {
            "input": "Write a Python function that returns Hello World.",
            "category": "Simple Code",
            "expected": "Should write basic function"
        },
        {
            "input": "What color is grass?",
            "category": "Simple Fact",
            "expected": "Should answer green"
        },
        {
            "input": "Count from 1 to 5.",
            "category": "Counting",
            "expected": "Should list 1, 2, 3, 4, 5"
        },
        {
            "input": "What is the opposite of hot?",
            "category": "Antonym",
            "expected": "Should answer cold"
        },
        {
            "input": "Translate 'hello' to Spanish.",
            "category": "Translation",
            "expected": "Should answer 'hola'"
        },
        {
            "input": "What comes after Monday?",
            "category": "Sequence",
            "expected": "Should answer Tuesday"
        }
    ]
    
    print("\nRunning deterministic tests...\n")
    
    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"[Test {i}/10] {test['category']}")
        print(f"Expected: {test['expected']}")
        print("-"*40)
        
        response = generate_response_cpu(
            model, tokenizer,
            test["input"],
            max_new_tokens=150,  # Shorter for simple tests
            temperature=0.1  # Very low for deterministic output
        )
        
        print(f"Response: {response}\n")
        
        # Simple quality check
        quality = "?"
        if test['category'] == 'Simple Math' and '4' in response:
            quality = "PASS"
        elif test['category'] == 'Factual' and 'Paris' in response.capitalize():
            quality = "PASS"
        elif test['category'] == 'Simple Fact' and 'green' in response.lower():
            quality = "PASS"
        
        results.append((test['category'], quality))
        time.sleep(0.2)  # Brief pause
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("-"*60)
    for category, quality in results:
        print(f"{quality} {category}")
    
    correct = sum(1 for _, q in results if q == "PASS")
    print(f"\nBasic Accuracy: {correct}/10 simple tests passed")
    print("="*60)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test SFT model on CPU with low temperature')
    parser.add_argument('--checkpoint', type=str, default='best_sft_model.pt',
                       help='Path to checkpoint')
    parser.add_argument('--quick', action='store_true', 
                       help='Run only 3 tests for quick check')
    
    args = parser.parse_args()
    
    # Check checkpoint
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint '{args.checkpoint}' not found!")
        return
    
    # Load model
    print("Loading model for CPU inference...")
    model, config = load_sft_model_cpu(args.checkpoint)
    tokenizer = get_tokenizer()
    
    print("\n[READY] Model loaded successfully!")
    print("Note: CPU inference will be slow (~5-10 tok/s)")
    print("Using temperature=0.1 for consistent outputs\n")
    
    if args.quick:
        # Quick test with 3 examples
        print("QUICK TEST MODE - 3 examples only\n")
        for test_input in [
            "What is 2 + 2?",
            "What color is the sky?",
            "Write hello in Python."
        ]:
            response = generate_response_cpu(
                model, tokenizer, test_input,
                max_new_tokens=50,
                temperature=0.1
            )
            print(f"Response: {response}\n")
            print("-"*40)
    else:
        # Full test suite
        run_quality_test_suite(model, tokenizer)

if __name__ == "__main__":
    main()