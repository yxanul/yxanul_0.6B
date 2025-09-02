#!/usr/bin/env python3
"""
Clean SFT Data Preparation with PROPER EOS tokens.
Handles both Alpaca (JSON) and GSM8K (Parquet) formats.
CRITICAL: Adds EOS after every assistant response so model learns to stop!
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
from tqdm import tqdm
import random

def load_tokenizer():
    """Load the SmolLM tokenizer used in pretraining."""
    print("Loading SmolLM2-135M tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        use_fast=True
    )
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  BOS token: {repr(tokenizer.bos_token)} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
    
    # Critical check
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have EOS token!")
    
    return tokenizer

def load_alpaca_data(file_path: str) -> List[Dict]:
    """Load and format Alpaca dataset."""
    print(f"\nLoading Alpaca data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} examples")
    
    formatted_data = []
    for item in data:
        # Format as conversation
        if item.get('input', '').strip():
            # Has input context
            user_content = f"{item['instruction']}\n\nInput: {item['input']}"
        else:
            # No input, just instruction
            user_content = item['instruction']
        
        formatted_data.append({
            'messages': [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': item['output']}
            ],
            'source': 'alpaca'
        })
    
    return formatted_data

def load_gsm8k_data(file_path: str) -> List[Dict]:
    """Load and format GSM8K dataset."""
    print(f"\nLoading GSM8K data from {file_path}")
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df)} examples")
    
    formatted_data = []
    for _, row in df.iterrows():
        # Format math problem as conversation
        formatted_data.append({
            'messages': [
                {'role': 'user', 'content': row['question']},
                {'role': 'assistant', 'content': row['answer']}
            ],
            'source': 'gsm8k'
        })
    
    return formatted_data

def create_training_example(messages: List[Dict], tokenizer, max_length: int = 2048) -> Dict:
    """
    Create a single training example with PROPER EOS tokens.
    
    CRITICAL CHANGES:
    1. Add EOS token after each assistant response
    2. Use BOS token at the start
    3. Mask user inputs (label=-100)
    4. Predict assistant responses INCLUDING the EOS token
    """
    
    # Start with BOS token if available
    if tokenizer.bos_token_id is not None and tokenizer.bos_token_id != tokenizer.eos_token_id:
        input_ids = [tokenizer.bos_token_id]
        labels = [-100]  # Don't predict BOS
    else:
        input_ids = []
        labels = []
    
    # Process each message
    for message in messages:
        role = message['role']
        content = message['content'].strip()
        
        if role == 'user':
            # Format and tokenize user input
            user_text = f"User: {content}\n"
            user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
            
            # Add to sequence
            input_ids.extend(user_tokens)
            labels.extend([-100] * len(user_tokens))  # Mask user input
            
        elif role == 'assistant':
            # Format and tokenize assistant response
            assistant_text = f"Assistant: {content}\n"
            assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
            
            # CRITICAL: Add EOS token after assistant response
            assistant_tokens.append(tokenizer.eos_token_id)
            
            # Add to sequence
            input_ids.extend(assistant_tokens)
            labels.extend(assistant_tokens)  # Predict assistant response INCLUDING EOS
    
    # Truncate if needed
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'length': len(input_ids)
    }

def process_datasets(
    alpaca_path: str = None,
    gsm8k_path: str = None,
    output_dir: str = 'data_sft_clean',
    max_examples: int = None,
    max_length: int = 2048,
    val_split: float = 0.05
):
    """Process both datasets and create training files."""
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Load datasets
    all_data = []
    
    if alpaca_path and Path(alpaca_path).exists():
        alpaca_data = load_alpaca_data(alpaca_path)
        all_data.extend(alpaca_data)
        print(f"  Added {len(alpaca_data)} Alpaca examples")
    
    if gsm8k_path and Path(gsm8k_path).exists():
        gsm8k_data = load_gsm8k_data(gsm8k_path)
        all_data.extend(gsm8k_data)
        print(f"  Added {len(gsm8k_data)} GSM8K examples")
    
    if not all_data:
        raise ValueError("No data loaded! Check file paths.")
    
    print(f"\nTotal examples loaded: {len(all_data)}")
    
    # Shuffle data
    random.seed(42)
    random.shuffle(all_data)
    
    # Limit examples if requested
    if max_examples and len(all_data) > max_examples:
        all_data = all_data[:max_examples]
        print(f"Limited to {max_examples} examples")
    
    # Process examples
    print(f"\nProcessing {len(all_data)} examples...")
    processed_examples = []
    total_tokens = 0
    
    for item in tqdm(all_data, desc="Tokenizing"):
        example = create_training_example(item['messages'], tokenizer, max_length)
        processed_examples.append(example)
        total_tokens += example['length']
    
    print(f"\nProcessing complete!")
    print(f"  Total examples: {len(processed_examples)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per example: {total_tokens/len(processed_examples):.1f}")
    
    # Check EOS token presence
    eos_count = sum(
        1 for ex in processed_examples 
        if tokenizer.eos_token_id in ex['input_ids']
    )
    print(f"  Examples with EOS token: {eos_count}/{len(processed_examples)} ({100*eos_count/len(processed_examples):.1f}%)")
    
    # Split into train/val
    split_idx = int((1 - val_split) * len(processed_examples))
    train_examples = processed_examples[:split_idx]
    val_examples = processed_examples[split_idx:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val: {len(val_examples)} examples")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save datasets
    def save_examples(examples, split_name):
        """Save examples as binary files."""
        all_input_ids = []
        all_labels = []
        
        for ex in examples:
            all_input_ids.extend(ex['input_ids'])
            all_labels.extend(ex['labels'])
        
        # Convert to numpy arrays
        input_array = np.array(all_input_ids, dtype=np.uint16)
        labels_array = np.array(all_labels, dtype=np.int32)  # int32 for -100
        
        # Save
        input_file = output_path / f"{split_name}_tokens.bin"
        labels_file = output_path / f"{split_name}_labels.bin"
        
        input_array.tofile(input_file)
        labels_array.tofile(labels_file)
        
        print(f"  Saved {split_name}: {len(all_input_ids):,} tokens")
        return len(all_input_ids)
    
    train_tokens = save_examples(train_examples, 'train')
    val_tokens = save_examples(val_examples, 'val')
    
    # Save metadata
    metadata = {
        'tokenizer': 'HuggingFaceTB/SmolLM2-135M',
        'max_length': max_length,
        'total_examples': len(processed_examples),
        'train_examples': len(train_examples),
        'val_examples': len(val_examples),
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'vocab_size': len(tokenizer),
        'eos_token_id': tokenizer.eos_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'sources': {
            'alpaca': alpaca_path is not None,
            'gsm8k': gsm8k_path is not None
        },
        'format': 'clean_sft_with_eos'
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SFT DATA PREPARATION COMPLETE")
    print('='*60)
    print(f"Output directory: {output_dir}")
    print(f"Format: User/Assistant with EOS tokens")
    print(f"Ready for training!")
    print(f"\nNext step:")
    print(f"python train_sft.py --data_dir {output_dir} --base_model ../train/best_model_fp8_optimized.pt")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Prepare clean SFT data with proper EOS tokens')
    parser.add_argument('--alpaca', type=str, default='alpaca_data_cleaned.json',
                       help='Path to Alpaca JSON file')
    parser.add_argument('--gsm8k', type=str, default='train-00000-of-00001.parquet',
                       help='Path to GSM8K parquet file')
    parser.add_argument('--output_dir', type=str, default='data_sft_clean',
                       help='Output directory')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum examples to use (None = all)')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--val_split', type=float, default=0.05,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Process datasets
    metadata = process_datasets(
        alpaca_path=args.alpaca if Path(args.alpaca).exists() else None,
        gsm8k_path=args.gsm8k if Path(args.gsm8k).exists() else None,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        max_length=args.max_length,
        val_split=args.val_split
    )
    
    print(f"\nDataset statistics:")
    print(f"  Training tokens: {metadata['train_tokens']:,}")
    print(f"  Expected iterations for 1 epoch at batch_size=8: {metadata['train_tokens'] // (8 * args.max_length)}")

if __name__ == "__main__":
    main()