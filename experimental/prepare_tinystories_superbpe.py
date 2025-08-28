#!/usr/bin/env python3
"""
Prepare TinyStories dataset using SuperBPE tokenizer.
Creates train.bin and val.bin with 40% fewer tokens!
"""

import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import json
from tqdm import tqdm

def prepare_dataset():
    """Prepare TinyStories with SuperBPE tokenizer."""
    
    # Load SuperBPE tokenizer from local cache
    print("Loading SuperBPE tokenizer from local cache...")
    
    # Check if we're on RunPod (Linux) or local Windows
    import os
    import platform
    
    # Determine cache path based on platform
    if platform.system() == "Linux":
        # RunPod instance paths
        cache_base = Path("/workspace/yxanul_0.6B/tokenizer_cache")
    else:
        # Local Windows paths
        cache_base = Path("D:/ai_testing/yxanul_0.6B/tokenizer_cache")
    
    # Check if cache exists
    if not cache_base.exists():
        # Try relative path from experimental directory
        cache_base = Path("../tokenizer_cache")
        if not cache_base.exists():
            raise FileNotFoundError(f"Tokenizer cache not found at {cache_base}")
    
    # Load SuperBPE-t80k tokenizer from local cache ONLY
    print("Loading SuperBPE-t80k tokenizer from local cache...")
    
    t80k_path = cache_base / "superbpe-t80k-fast"
    if not t80k_path.exists():
        raise FileNotFoundError(f"SuperBPE-t80k cache not found at {t80k_path}")
    
    print(f"Found t80k cache at {t80k_path}")
    
    # Load using AutoTokenizer with use_fast=True for performance
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(t80k_path),
            use_fast=True,  # We need fast tokenizer for performance
            local_files_only=True
        )
        print(f"SUCCESS: Loaded SuperBPE-t80k fast tokenizer!")
    except Exception as e:
        print(f"Failed to load with use_fast=True: {e}")
        print("ERROR: Could not load SuperBPE-t80k tokenizer")
        print("Please ensure you have a compatible transformers version installed")
        raise RuntimeError(f"Failed to load SuperBPE-t80k tokenizer: {e}")
    
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Paths - look for existing TinyStories binary files
    # Check for existing GPT-2 tokenized data first
    gpt2_train = Path('data/train.bin')
    gpt2_val = Path('data/val.bin')
    
    if gpt2_train.exists() and gpt2_val.exists():
        print("\nFound existing GPT-2 tokenized data in data/ directory")
        print("But we need the original text files to re-tokenize with SuperBPE")
    
    # Look for TinyStories parquet files
    data_dir = None
    possible_locations = [
        Path('TinyStories-hf/data'),  # Current location in experimental/
        Path('TinyStories-hf'),  # Without data subdirectory
        Path('../TinyStories'),  # Parent directory
        Path('../TinyStories-hf'),  # HuggingFace format in parent
        Path('/workspace/TinyStories'),  # Absolute path on RunPod
    ]
    
    for location in possible_locations:
        if location.exists():
            # Check if it has parquet files
            if list(location.glob('*.parquet')):
                data_dir = location
                print(f"Found TinyStories dataset at: {data_dir}")
                break
    
    if data_dir is None:
        print("\nERROR: Cannot find TinyStories dataset!")
        print("Please download TinyStories dataset first:")
        print("  1. Download from HuggingFace: https://huggingface.co/datasets/roneneldan/TinyStories")
        print("  2. Extract parquet files to one of these locations:")
        for loc in possible_locations:
            print(f"     - {loc}")
        raise FileNotFoundError("TinyStories dataset not found - see instructions above")
    
    output_dir = Path('data_superbpe')
    output_dir.mkdir(exist_ok=True)
    
    # Process training data
    print("\nProcessing training data...")
    train_files = list(data_dir.glob('train*.parquet'))
    
    if not train_files:
        print("No training files found! Looking for alternative structure...")
        # Try loading from single file
        train_file = data_dir / 'train.parquet'
        if train_file.exists():
            train_files = [train_file]
        else:
            raise FileNotFoundError(f"No training data found in {data_dir}")
    
    all_train_tokens = []
    total_stories = 0
    
    for file in tqdm(train_files, desc="Processing train files"):
        df = pd.read_parquet(file)
        
        # TinyStories has 'text' column
        texts = df['text'].tolist()
        total_stories += len(texts)
        
        # Tokenize in batches for efficiency
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Join stories with a separator (like original)
            batch_text = ' '.join(batch)
            tokens = tokenizer.encode(batch_text, add_special_tokens=False)
            all_train_tokens.extend(tokens)
    
    train_tokens = np.array(all_train_tokens, dtype=np.uint32)
    print(f"Train tokens: {len(train_tokens):,} from {total_stories:,} stories")
    
    # Save training data
    train_output = output_dir / 'train.bin'
    train_tokens.astype(np.uint16).tofile(train_output)
    print(f"Saved to {train_output}")
    
    # Process validation data
    print("\nProcessing validation data...")
    val_files = list(data_dir.glob('val*.parquet'))
    
    if not val_files:
        val_file = data_dir / 'validation.parquet'
        if val_file.exists():
            val_files = [val_file]
        else:
            print("No validation data found, using 10% of training data")
            # Use last 10% of training tokens as validation
            split_idx = int(len(train_tokens) * 0.9)
            val_tokens = train_tokens[split_idx:]
            train_tokens = train_tokens[:split_idx]
    else:
        all_val_tokens = []
        total_val_stories = 0
        
        for file in tqdm(val_files, desc="Processing val files"):
            df = pd.read_parquet(file)
            texts = df['text'].tolist()
            total_val_stories += len(texts)
            
            # Tokenize in batches
            batch_size = 1000
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_text = ' '.join(batch)
                tokens = tokenizer.encode(batch_text, add_special_tokens=False)
                all_val_tokens.extend(tokens)
        
        val_tokens = np.array(all_val_tokens, dtype=np.uint32)
        print(f"Val tokens: {len(val_tokens):,} from {total_val_stories:,} stories")
    
    # Save validation data
    val_output = output_dir / 'val.bin'
    val_tokens.astype(np.uint16).tofile(val_output)
    print(f"Saved to {val_output}")
    
    # Save tokenizer config for reference
    config = {
        'tokenizer_name': tokenizer.name_or_path,
        'vocab_size': len(tokenizer),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'total_stories': total_stories,
    }
    
    with open(output_dir / 'tokenizer_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Compare with GPT-2 if available
    gpt2_train = Path('data/train.bin')
    if gpt2_train.exists():
        gpt2_size = gpt2_train.stat().st_size / 2  # uint16 = 2 bytes
        superbpe_size = len(train_tokens)
        reduction = (1 - superbpe_size / gpt2_size) * 100
        
        print("\n" + "="*60)
        print("TOKENIZATION COMPARISON")
        print("="*60)
        print(f"GPT-2 tokens: {int(gpt2_size):,}")
        print(f"SuperBPE tokens: {superbpe_size:,}")
        print(f"Reduction: {reduction:.1f}%")
        print(f"Speedup factor: {gpt2_size/superbpe_size:.2f}x")
    
    print("\n" + "="*60)
    print("DATASET READY!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print("\nTo use this dataset, update train_tinystories.py:")
    print("  1. Change data_dir to 'data_superbpe'")
    print(f"  2. Set vocab_size to {len(tokenizer)}")
    print("  3. Adjust model embedding size if needed")
    print("\nExpected benefits:")
    print("  - 40% faster training")
    print("  - 40% longer effective context")
    print("  - Better compression of common phrases")

if __name__ == "__main__":
    prepare_dataset()