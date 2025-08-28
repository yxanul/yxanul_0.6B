#!/usr/bin/env python3
"""
Prepare TinyStories dataset for training.
Downloads and converts to memory-mapped format for fast loading.
"""

import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import tiktoken

def main():
    # Configuration
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Load tokenizer (GPT-2 tokenizer for compatibility)
    print("Loading tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    
    # Load TinyStories dataset
    print("Loading TinyStories dataset...")
    try:
        # Try the standard way first
        dataset = load_dataset("roneneldan/TinyStories")
    except ValueError as e:
        if "Invalid pattern" in str(e):
            # Fallback for newer datasets library versions
            print("Using alternative loading method due to datasets library version...")
            dataset = load_dataset("roneneldan/TinyStories", data_files={"train": "*.parquet", "validation": "*.parquet"})
        else:
            raise
    
    # Process each split
    for split in ['train', 'validation']:
        print(f"\nProcessing {split} split...")
        
        # Get the data
        if split == 'validation':
            # TinyStories has 'validation' split
            data = dataset['validation']
        else:
            data = dataset['train']
        
        # Tokenize all texts
        print(f"Tokenizing {len(data)} examples...")
        all_tokens = []
        
        for example in tqdm(data, desc=f"Tokenizing {split}"):
            text = example['text']
            tokens = enc.encode_ordinary(text)
            all_tokens.extend(tokens)
            # Add EOS token
            all_tokens.append(enc.eot_token)
        
        # Convert to numpy array
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        
        # Save as memory-mapped file
        output_file = output_dir / f"{split}.bin"
        print(f"Writing {len(all_tokens):,} tokens to {output_file}")
        
        # Create memory-mapped file
        arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(len(all_tokens),))
        arr[:] = all_tokens
        arr.flush()
        
        print(f"{split}: {len(all_tokens):,} tokens saved")
    
    # Save metadata
    metadata = {
        'vocab_size': enc.n_vocab,
        'train_tokens': len(np.memmap(output_dir / 'train.bin', dtype=np.uint16, mode='r')),
        'val_tokens': len(np.memmap(output_dir / 'validation.bin', dtype=np.uint16, mode='r')),
    }
    
    print("\nDataset preparation complete!")
    print(f"Train tokens: {metadata['train_tokens']:,}")
    print(f"Validation tokens: {metadata['val_tokens']:,}")
    print(f"Total tokens: {metadata['train_tokens'] + metadata['val_tokens']:,}")

if __name__ == "__main__":
    main()