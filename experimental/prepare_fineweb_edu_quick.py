#!/usr/bin/env python3
"""
Prepare mixed-domain dataset for training.
Processes half of the 3B token mixed dataset (~1.5B tokens).
Dataset includes diverse content: text, code, math, etc.
Memory-efficient version with streaming to disk.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import platform
import tempfile
import os

def tokenize_batch_fast(texts, tokenizer, eos_token_id):
    """Fast batch tokenization without multiprocessing overhead."""
    all_tokens = []
    
    # Filter out empty texts
    valid_texts = [text for text in texts if pd.notna(text) and len(text.strip()) > 0]
    
    if not valid_texts:
        return all_tokens
    
    # Batch encode all texts at once - MUCH faster than individual encoding
    batch_encodings = tokenizer(
        valid_texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )
    
    # Add each encoded text with EOS separator
    for tokens in batch_encodings['input_ids']:
        all_tokens.extend(tokens)
        all_tokens.append(eos_token_id)
    
    return all_tokens

def prepare_dataset_quick():
    """
    Prepare mixed-domain dataset (half of 3B tokens).
    Memory-efficient version that streams to disk.
    """
    
    print("="*60)
    print("MIXED-DOMAIN DATASET PREPARATION")
    print("Processing 1.5B tokens from mixed-pretrain-3b")
    print("With SmolLM2-135M Tokenizer")
    print("Memory-efficient streaming version")
    print("="*60)
    
    # Load SmolLM2 tokenizer
    print("\nLoading SmolLM2-135M tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            use_fast=True
        )
        print(f"SUCCESS: Loaded SmolLM2-135M tokenizer!")
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/SmolLM-1.7B",
                use_fast=True
            )
            print(f"Loaded SmolLM v1 tokenizer as fallback")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Get EOS token ID
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    print(f"EOS token ID: {eos_token_id}")
    
    # Determine data path based on platform
    if platform.system() == "Linux":
        # RunPod instance paths
        data_file = Path("/workspace/yxanul_0.6B/experimental/mixed-pretrain-3b/mixed_dataset_3b.parquet")
    else:
        # Local Windows paths
        data_file = Path("D:/ai_testing/yxanul_0.6B/experimental/mixed-pretrain-3b/mixed_dataset_3b.parquet")
    
    # Check for dataset file
    if not data_file.exists():
        # Try relative path
        data_file = Path("mixed-pretrain-3b/mixed_dataset_3b.parquet")
        if not data_file.exists():
            print(f"\nERROR: Cannot find dataset at {data_file}")
            print("Please download from: https://huggingface.co/datasets/Yxanul/mixed-pretrain-3b")
            print("Place in: experimental/mixed-pretrain-3b/mixed_dataset_3b.parquet")
            raise FileNotFoundError("Dataset not found")
    
    # Load the single parquet file
    print(f"\nLoading mixed dataset from {data_file.name}...")
    print("This dataset contains diverse content: text, code, math, etc.")
    
    # Create output directory
    output_dir = Path('data_mixed_3b')
    output_dir.mkdir(exist_ok=True)
    
    # Create temporary file for streaming tokens
    temp_tokens_file = output_dir / 'temp_tokens.bin'
    
    # Load and process the parquet file
    print("\nLoading parquet file...")
    df = pd.read_parquet(data_file)
    
    # Check columns
    if 'text' not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        # Try 'content' or other common column names
        if 'content' in df.columns:
            df['text'] = df['content']
        else:
            raise ValueError("No 'text' or 'content' column found in dataset")
    
    total_docs = len(df)
    print(f"Total documents: {total_docs:,}")
    
    # Process FULL dataset for proper training
    fraction = 1.0  # Use entire dataset for real training
    num_docs = int(total_docs * fraction)
    print(f"Processing {num_docs:,} documents ({fraction*100:.0f}% of dataset)")
    df = df.head(num_docs)
    
    texts = df['text'].tolist()
    del df  # Free memory immediately
    
    total_chars = 0
    total_tokens = 0
    
    # Process in smaller batches and stream to disk
    print("\nTokenizing documents (memory-efficient mode)...")
    batch_size = 2500  # Smaller batch size to reduce memory
    chunk_size = 100_000_000  # Write to disk every 100M tokens
    current_chunk = []
    
    with open(temp_tokens_file, 'wb') as f:
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch = texts[i:min(i+batch_size, len(texts))]
            
            # Track character count
            for text in batch:
                if pd.notna(text):
                    total_chars += len(text)
            
            # Tokenize batch
            tokens = tokenize_batch_fast(batch, tokenizer, eos_token_id)
            current_chunk.extend(tokens)
            
            # Write chunk to disk when it gets large
            if len(current_chunk) >= chunk_size:
                # Convert to numpy and write
                chunk_array = np.array(current_chunk, dtype=np.uint16)
                chunk_array.tofile(f)
                total_tokens += len(current_chunk)
                print(f"  Written {total_tokens:,} tokens to disk...")
                current_chunk = []  # Clear memory
        
        # Write remaining tokens
        if current_chunk:
            chunk_array = np.array(current_chunk, dtype=np.uint16)
            chunk_array.tofile(f)
            total_tokens += len(current_chunk)
    
    print(f"\nTotal tokens written: {total_tokens:,}")
    
    # Now read back and split into train/val
    print("\nSplitting into train/val sets...")
    split_idx = int(total_tokens * 0.95)
    
    # Read and write train set
    print("Writing training set...")
    train_output = output_dir / 'train.bin'
    with open(temp_tokens_file, 'rb') as infile:
        with open(train_output, 'wb') as outfile:
            # Copy first 95% of tokens
            bytes_to_copy = split_idx * 2  # 2 bytes per uint16
            chunk_size = 100 * 1024 * 1024  # 100MB chunks
            copied = 0
            while copied < bytes_to_copy:
                to_read = min(chunk_size, bytes_to_copy - copied)
                data = infile.read(to_read)
                if not data:
                    break
                outfile.write(data)
                copied += len(data)
    
    print("Writing validation set...")
    val_output = output_dir / 'val.bin'
    with open(temp_tokens_file, 'rb') as infile:
        infile.seek(split_idx * 2)  # Skip training data
        with open(val_output, 'wb') as outfile:
            # Copy remaining tokens
            while True:
                data = infile.read(chunk_size)
                if not data:
                    break
                outfile.write(data)
    
    # Clean up temp file
    os.remove(temp_tokens_file)
    
    # Calculate final sizes
    train_tokens = split_idx
    val_tokens = total_tokens - split_idx
    
    print(f"\nDataset statistics:")
    print(f"  Documents: {num_docs:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total characters: {total_chars:,}")
    if num_docs > 0:
        print(f"  Avg tokens/doc: {total_tokens/num_docs:.1f}")
        print(f"  Avg chars/token: {total_chars/total_tokens:.2f}")
    
    print(f"\nFinal split:")
    print(f"  Train: {train_tokens:,} tokens")
    print(f"  Val: {val_tokens:,} tokens")
    
    # Save config
    config = {
        'dataset': 'mixed-pretrain-3b',
        'mode': f'{fraction:.0%}_dataset',
        'tokenizer': 'HuggingFaceTB/SmolLM2-135M',
        'vocab_size': len(tokenizer),
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'total_documents': num_docs,
        'dataset_type': 'mixed-domain (text, code, math)',
    }
    
    with open(output_dir / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("MIXED DATASET READY!")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Size: {total_tokens/1e9:.1f}B tokens")
    print(f"Content: Mixed-domain (text, code, math)")
    print("\nTo test FP8 on RTX 5090:")
    print("  # Run FP8 training:")
    print("  python train_fp8.py --data_dir data_mixed_3b \\")
    print("    --max_iters 1000 --eval_interval 100 --compile")
    print("\n  # Compare with BF16:")
    print("  python train_fp8.py --data_dir data_mixed_3b \\")
    print("    --max_iters 1000 --eval_interval 100 --no_fp8 --compile")
    print("\nWatch for tokens/sec to see if FP8 gives speedup!")

if __name__ == "__main__":
    prepare_dataset_quick()