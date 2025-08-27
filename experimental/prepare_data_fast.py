#!/usr/bin/env python3
"""
Fast data preparation - no redundant counting.
We know it's ~1B tokens, just tokenize and write directly.
"""

import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def prepare_dataset_fast(
    parquet_path="../experimental-pretrain-1b/dataset_1b.parquet",
    tokenizer_name="gpt2",
    max_length=2048,
    output_dir="data",
    val_split=0.05,
    estimate_tokens=1_000_000_000  # We know it's ~1B
):
    """Fast conversion - no counting, just process."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    if tokenizer_name == "superbpe":
        tokenizer = AutoTokenizer.from_pretrained("../tokenizer_cache/superbpe-t80k-fast")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Vocab size: {len(tokenizer)}")
    
    # Load dataset
    print(f"Loading dataset...")
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Check if dataset has precomputed token counts
    if 'num_tokens' in dataset.features:
        total_tokens = sum(dataset['num_tokens'])
        print(f"Dataset has precomputed token count: {total_tokens:,}")
        estimate_tokens = total_tokens
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")
    
    # Process each split
    for split_name, split_data, size_estimate in [
        ("train", train_dataset, int(estimate_tokens * 0.95)),
        ("val", val_dataset, int(estimate_tokens * 0.05))
    ]:
        print(f"\nProcessing {split_name} split...")
        
        # Pre-allocate array with estimated size (we'll resize at the end)
        filename = Path(output_dir) / f"{split_name}.bin"
        dtype = np.uint32 if len(tokenizer) > 65535 else np.uint16
        
        # Allocate with some buffer
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(int(size_estimate * 1.1),))
        
        # Tokenize and write directly (no separate counting phase)
        idx = 0
        batch_size = 1000  # Process in batches for progress bar
        
        with tqdm(total=len(split_data), desc=f"Tokenizing {split_name}") as pbar:
            for i in range(0, len(split_data), batch_size):
                batch = split_data[i:min(i+batch_size, len(split_data))]
                
                # Process batch
                for example in batch['text']:
                    ids = tokenizer.encode(example, max_length=max_length, truncation=True)
                    
                    # Check if we need to resize
                    if idx + len(ids) > len(arr):
                        # Extend array
                        arr._mmap.resize(int(len(arr) * 1.5))
                        arr = np.memmap(filename, dtype=dtype, mode='r+', shape=(int(len(arr) * 1.5),))
                    
                    arr[idx:idx+len(ids)] = ids
                    idx += len(ids)
                
                pbar.update(len(batch['text']))
        
        # Trim to actual size
        arr._mmap.resize(idx * dtype.itemsize)
        arr = np.memmap(filename, dtype=dtype, mode='r+', shape=(idx,))
        arr.flush()
        
        print(f"Saved {filename}")
        print(f"  Tokens: {idx:,}")
        print(f"  Size: {filename.stat().st_size / 1e9:.2f} GB")
    
    # Save metadata
    metadata = {
        'vocab_size': len(tokenizer),
        'tokenizer': tokenizer_name,
        'max_length': max_length,
        'dataset_examples': len(dataset),
    }
    
    import json
    with open(Path(output_dir) / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nData preparation complete!")
    return metadata


def prepare_dataset_parallel(
    parquet_path="../experimental-pretrain-1b/dataset_1b.parquet",
    tokenizer_name="gpt2",
    max_length=2048,
    output_dir="data",
    val_split=0.05,
    num_workers=8
):
    """Even faster with parallel processing."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    if tokenizer_name == "superbpe":
        tokenizer = AutoTokenizer.from_pretrained("../tokenizer_cache/superbpe-t80k-fast")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset with num_proc for faster loading
    print(f"Loading dataset with {num_workers} workers...")
    dataset = load_dataset(
        "parquet", 
        data_files=parquet_path, 
        split="train",
        num_proc=num_workers
    )
    
    # Tokenize with multiple workers
    def tokenize_function(examples):
        return {
            'input_ids': [
                tokenizer.encode(text, max_length=max_length, truncation=True)
                for text in examples['text']
            ]
        }
    
    print(f"Tokenizing {len(dataset)} examples with {num_workers} workers...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=['text', 'source', 'num_tokens'] if 'num_tokens' in dataset.features else ['text'],
        num_proc=num_workers,
        desc="Tokenizing"
    )
    
    # Split
    val_size = int(len(tokenized) * val_split)
    train_data = tokenized.select(range(len(tokenized) - val_size))
    val_data = tokenized.select(range(len(tokenized) - val_size, len(tokenized)))
    
    # Save to memmap
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        print(f"\nWriting {split_name} split...")
        
        # Count total tokens
        total_tokens = sum(len(ids) for ids in split_data['input_ids'])
        print(f"  Total tokens: {total_tokens:,}")
        
        # Create memmap
        filename = Path(output_dir) / f"{split_name}.bin"
        dtype = np.uint32 if len(tokenizer) > 65535 else np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_tokens,))
        
        # Write tokens
        idx = 0
        for ids in tqdm(split_data['input_ids'], desc=f"Writing {split_name}.bin"):
            arr[idx:idx+len(ids)] = ids
            idx += len(ids)
        
        arr.flush()
        print(f"  Saved: {filename.stat().st_size / 1e9:.2f} GB")
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for parallel')
    parser.add_argument('--tokenizer', default='gpt2', help='Tokenizer to use')
    args = parser.parse_args()
    
    if args.parallel:
        # Use parallel version (fastest)
        prepare_dataset_parallel(
            tokenizer_name=args.tokenizer,
            num_workers=args.workers
        )
    else:
        # Use fast sequential version
        prepare_dataset_fast(
            tokenizer_name=args.tokenizer
        )