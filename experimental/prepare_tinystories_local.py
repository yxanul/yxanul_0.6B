#!/usr/bin/env python3
"""
Prepare TinyStories dataset from locally cloned repository.
Reads parquet files from TinyStories-hf/data/ directory.
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import tiktoken

def main():
    # Configuration
    data_source_dir = Path("TinyStories-hf/data")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    print("TinyStories Local Dataset Preparation")
    print("=" * 60)
    
    # Check if source directory exists
    if not data_source_dir.exists():
        print(f"Error: {data_source_dir} not found!")
        print("Please clone the repository first:")
        print("  git clone https://huggingface.co/datasets/skeskinen/TinyStories-hf")
        sys.exit(1)
    
    # Load tokenizer
    print("Loading GPT-2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    
    # Find parquet files
    print(f"\nLooking for parquet files in {data_source_dir}...")
    parquet_files = list(data_source_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("No parquet files found in data directory!")
        print("Contents of directory:")
        for f in data_source_dir.iterdir():
            print(f"  - {f.name}")
        sys.exit(1)
    
    print(f"Found {len(parquet_files)} parquet files:")
    for f in parquet_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    # Process files by split
    splits_data = {'train': [], 'validation': []}
    
    for pf in parquet_files:
        # Determine split from filename
        filename_lower = pf.name.lower()
        if 'train' in filename_lower:
            split = 'train'
        elif 'valid' in filename_lower or 'val' in filename_lower:
            split = 'validation'
        else:
            # Default to train if unclear
            split = 'train'
            print(f"Warning: Could not determine split for {pf.name}, assuming train")
        
        print(f"\nLoading {pf.name} for {split} split...")
        df = pd.read_parquet(pf)
        print(f"  Loaded {len(df)} stories")
        
        # Find text column
        text_col = None
        for col in ['text', 'story', 'content']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            print(f"  Available columns: {df.columns.tolist()}")
            # Try first column if it's string type
            if df.dtypes[df.columns[0]] == 'object':
                text_col = df.columns[0]
                print(f"  Using column '{text_col}' as text")
            else:
                print(f"  Error: No text column found in {pf.name}")
                continue
        
        # Extract texts
        texts = df[text_col].dropna().tolist()
        splits_data[split].extend(texts)
        print(f"  Added {len(texts)} non-empty stories to {split}")
    
    # Tokenize and save each split
    for split_name, texts in splits_data.items():
        if not texts:
            print(f"\nWarning: No data for {split_name} split, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split: {len(texts)} stories")
        print(f"{'='*60}")
        
        # Tokenize all texts
        all_tokens = []
        
        for text in tqdm(texts, desc=f"Tokenizing {split_name}"):
            if text and isinstance(text, str):
                tokens = enc.encode_ordinary(text)
                all_tokens.extend(tokens)
                # Add EOS token
                all_tokens.append(enc.eot_token)
        
        # Convert to numpy array
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        
        # Save as memory-mapped file
        # Use 'val.bin' for validation to match train_tinystories.py expectations
        if split_name == 'validation':
            output_filename = 'val.bin'
        else:
            output_filename = f'{split_name}.bin'
        
        output_file = output_dir / output_filename
        print(f"Writing {len(all_tokens):,} tokens to {output_file}")
        
        # Create memory-mapped file
        arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(len(all_tokens),))
        arr[:] = all_tokens
        arr.flush()
        
        print(f"✓ {split_name}: {len(all_tokens):,} tokens saved")
    
    # Create metadata file
    import json
    metadata = {
        'vocab_size': enc.n_vocab,
        'tokenizer': 'tiktoken-gpt2',
        'train_tokens': 0,
        'val_tokens': 0
    }
    
    # Check what files were created
    train_file = output_dir / 'train.bin'
    val_file = output_dir / 'val.bin'
    
    if train_file.exists():
        metadata['train_tokens'] = len(np.memmap(train_file, dtype=np.uint16, mode='r'))
    
    if val_file.exists():
        metadata['val_tokens'] = len(np.memmap(val_file, dtype=np.uint16, mode='r'))
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"Train tokens: {metadata['train_tokens']:,}")
    print(f"Val tokens: {metadata['val_tokens']:,}")
    print(f"Total tokens: {metadata['train_tokens'] + metadata['val_tokens']:,}")
    print("\nFiles created:")
    for f in output_dir.glob('*.bin'):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    print(f"  - metadata.json")
    print("\nReady for training! Run:")
    print("  python train_tinystories.py --dtype bfloat16 --max_iters 10000")

if __name__ == "__main__":
    main()