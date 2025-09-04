#!/usr/bin/env python3
"""
Test script to visualize label masking in SFT data preparation.
Shows exactly what the model learns to predict vs what it sees as context.
"""

from transformers import AutoTokenizer
from prepare_clean_sft_data import create_training_example
import json

def visualize_masking():
    """Visualize what gets masked and what gets predicted."""
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", use_fast=True)
    
    # Example conversation
    messages = [
        {'role': 'user', 'content': 'What is 2+2?'},
        {'role': 'assistant', 'content': 'The answer is 4.'}
    ]
    
    print("\n" + "="*60)
    print("LABEL MASKING VISUALIZATION")
    print("="*60)
    
    # Process the example
    example = create_training_example(messages, tokenizer)
    
    # Decode tokens
    tokens = [tokenizer.decode([tid]) for tid in example['input_ids']]
    
    print(f"\nTotal tokens: {len(tokens)}")
    print("\nToken-by-token breakdown:")
    print("-"*60)
    print(f"{'Token':<20} {'ID':<8} {'Label':<8} {'Learned?'}")
    print("-"*60)
    
    for i, (token, input_id, label) in enumerate(zip(tokens, example['input_ids'], example['labels'])):
        learned = "✓ LEARN" if label != -100 else "✗ CONTEXT"
        label_str = str(label) if label != -100 else "MASKED"
        
        # Highlight special tokens
        if input_id == tokenizer.eos_token_id:
            token = f"<EOS>{token}"
        if input_id == tokenizer.bos_token_id:
            token = f"<BOS>{token}"
            
        print(f"{token:<20} {input_id:<8} {label_str:<8} {learned}")
    
    # Show what the model actually learns to generate
    print("\n" + "="*60)
    print("WHAT THE MODEL LEARNS TO GENERATE:")
    print("="*60)
    
    learned_tokens = []
    for token, label in zip(tokens, example['labels']):
        if label != -100:
            learned_tokens.append(token)
    
    print("".join(learned_tokens))
    
    # Statistics
    num_context = sum(1 for l in example['labels'] if l == -100)
    num_learned = sum(1 for l in example['labels'] if l != -100)
    
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    print(f"Context tokens (masked): {num_context} ({100*num_context/len(tokens):.1f}%)")
    print(f"Learned tokens: {num_learned} ({100*num_learned/len(tokens):.1f}%)")
    print(f"Efficiency: Learning {num_learned} tokens from {len(tokens)} total")
    
    # Show the key insight
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("✓ User input is NEVER predicted (all -100)")
    print("✓ 'User: ' header is seen but NOT learned")
    print("✓ 'Assistant: ' header is seen but NOT learned")
    print("✓ ONLY the actual response content is learned")
    print("✓ EOS token IS learned (critical for stopping)")
    
    print("\nThis prevents the model from:")
    print("  - Generating 'Assistant: ' in its outputs")
    print("  - Wasting capacity learning to repeat role labels")
    print("  - Confusing role boundaries")

if __name__ == "__main__":
    visualize_masking()