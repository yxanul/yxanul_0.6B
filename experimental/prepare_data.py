#!/usr/bin/env python3
"""
Simple data preparation script - converts parquet to memmap for fast training.
Inspired by nanoGPT's approach - simple, fast, battle-tested.
"""

import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def prepare_dataset(
    parquet_path="../experimental-pretrain-1b/dataset_1b.parquet",  # Go up one level to root
    tokenizer_name="gpt2",  # Start with GPT2's 50k vocab for testing
    max_length=2048,
    output_dir="data",  # Save in experimental/data (we're already in experimental/)
    val_split=0.05
):
    """Convert parquet dataset to memmap files for fast training."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    if tokenizer_name == "superbpe":
        # Use the cached SuperBPE tokenizer (at root level)
        tokenizer = AutoTokenizer.from_pretrained("../tokenizer_cache/superbpe-t80k-fast")
        print(f"Loaded SuperBPE with {len(tokenizer)} vocab size")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Loaded {tokenizer_name} with {len(tokenizer)} vocab size")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset from {parquet_path}")
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")
    
    # Tokenize and save
    for split_name, split_data in [("train", train_dataset), ("val", val_dataset)]:
        print(f"\nProcessing {split_name} split...")
        
        # Count total tokens first
        total_tokens = 0
        for example in tqdm(split_data, desc="Counting tokens"):
            ids = tokenizer.encode(example['text'], max_length=max_length, truncation=True)
            total_tokens += len(ids)
        
        print(f"Total tokens in {split_name}: {total_tokens:,}")
        
        # Create memmap file
        filename = Path(output_dir) / f"{split_name}.bin"
        dtype = np.uint32 if len(tokenizer) > 65535 else np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_tokens,))
        
        # Write tokens
        idx = 0
        for example in tqdm(split_data, desc=f"Writing {split_name}.bin"):
            ids = tokenizer.encode(example['text'], max_length=max_length, truncation=True)
            arr[idx:idx+len(ids)] = ids
            idx += len(ids)
        
        arr.flush()
        print(f"Saved {filename} ({total_tokens:,} tokens, {filename.stat().st_size / 1e9:.2f} GB)")
    
    # Save metadata
    metadata = {
        'vocab_size': len(tokenizer),
        'tokenizer': tokenizer_name,
        'max_length': max_length,
        'train_tokens': total_tokens,
        'val_tokens': val_size,
    }
    
    import json
    with open(Path(output_dir) / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nData preparation complete!")
    return metadata

if __name__ == "__main__":
    # Prepare with GPT-2 tokenizer first (50k vocab, much easier on memory)
    metadata = prepare_dataset(
        tokenizer_name="gpt2",
        max_length=2048,
        output_dir="data"  # Will create data/ folder in experimental/
    )
    
    print(f"\nDataset ready for training:")
    print(f"  Vocab size: {metadata['vocab_size']}")
    print(f"  Train tokens: {metadata['train_tokens']:,}")
    print(f"  Val tokens: {metadata['val_tokens']:,}")