#!/usr/bin/env python3
"""
Prepare tiny-textbooks dataset using SuperBPE tokenizer.
Creates train.bin and val.bin with proper EOS tokens between documents.
Uses the 'textbook' column from the parquet files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import json
from tqdm import tqdm

def prepare_dataset():
    """Prepare tiny-textbooks with SuperBPE tokenizer."""
    
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
    
    # Get EOS token ID for proper document separation
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # Fallback to common EOS token
        eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    print(f"EOS token ID: {eos_token_id}")
    
    # Look for tiny-textbooks dataset
    data_dir = None
    possible_locations = [
        Path('tiny-textbooks/tiny-textbooks'),  # Nested structure as described
        Path('tiny-textbooks'),  # Single level
        Path('../tiny-textbooks/tiny-textbooks'),  # Parent directory
        Path('/workspace/tiny-textbooks/tiny-textbooks'),  # Absolute path on RunPod
    ]
    
    for location in possible_locations:
        if location.exists():
            # Check if it has the expected parquet file
            train_file = location / 'train-00000-of-00001.parquet'
            if train_file.exists():
                data_dir = location
                print(f"Found tiny-textbooks dataset at: {data_dir}")
                break
    
    if data_dir is None:
        print("\nERROR: Cannot find tiny-textbooks dataset!")
        print("Please download tiny-textbooks dataset first:")
        print("  1. Download from HuggingFace: https://huggingface.co/datasets/nampdn-ai/tiny-textbooks")
        print("  2. Place train-00000-of-00001.parquet in one of these locations:")
        for loc in possible_locations:
            print(f"     - {loc}")
        raise FileNotFoundError("tiny-textbooks dataset not found - see instructions above")
    
    output_dir = Path('data_textbooks_superbpe')
    output_dir.mkdir(exist_ok=True)
    
    # Process training data
    print("\nProcessing training data...")
    train_file = data_dir / 'train-00000-of-00001.parquet'
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    print(f"Loading {train_file}...")
    df = pd.read_parquet(train_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Use the 'textbook' column as specified
    if 'textbook' not in df.columns:
        raise ValueError(f"'textbook' column not found! Available columns: {df.columns.tolist()}")
    
    texts = df['textbook'].tolist()
    print(f"Processing {len(texts):,} textbook documents...")
    
    # Tokenize each document and add EOS token between them
    all_train_tokens = []
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing documents"):
        batch = texts[i:min(i+batch_size, len(texts))]
        
        for text in batch:
            if pd.isna(text) or len(text.strip()) == 0:
                continue
            
            # Tokenize the document
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Add tokens and EOS separator
            all_train_tokens.extend(tokens)
            all_train_tokens.append(eos_token_id)
    
    train_tokens = np.array(all_train_tokens, dtype=np.uint32)
    print(f"Total train tokens: {len(train_tokens):,} from {len(texts):,} documents")
    
    # Calculate average tokens per document
    avg_tokens = len(train_tokens) / len(texts)
    print(f"Average tokens per document: {avg_tokens:.1f}")
    
    # Split into train/val (90/10 split)
    split_idx = int(len(train_tokens) * 0.9)
    val_tokens = train_tokens[split_idx:]
    train_tokens = train_tokens[:split_idx]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Save training data
    train_output = output_dir / 'train.bin'
    train_tokens.astype(np.uint16).tofile(train_output)
    print(f"Saved training data to {train_output}")
    
    # Save validation data
    val_output = output_dir / 'val.bin'
    val_tokens.astype(np.uint16).tofile(val_output)
    print(f"Saved validation data to {val_output}")
    
    # Save tokenizer config for reference
    config = {
        'dataset': 'tiny-textbooks',
        'tokenizer_name': 'superbpe-t80k',
        'vocab_size': len(tokenizer),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'total_documents': len(texts),
        'avg_tokens_per_doc': avg_tokens,
        'eos_token_id': eos_token_id,
    }
    
    with open(output_dir / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Estimate with GPT-2 tokenizer for comparison
    print("\n" + "="*60)
    print("ESTIMATING GPT-2 TOKENIZATION")
    print("="*60)
    
    # Sample 100 documents for estimation
    sample_size = min(100, len(texts))
    sample_texts = texts[:sample_size]
    
    # Count characters
    total_chars = sum(len(text) for text in sample_texts if pd.notna(text))
    chars_per_superbpe = total_chars / sum(len(tokenizer.encode(text, add_special_tokens=False)) 
                                           for text in sample_texts if pd.notna(text))
    
    # GPT-2 typically has ~4.3 chars/token on English text
    gpt2_chars_per_token = 4.3
    
    # Estimate reduction
    estimated_gpt2_tokens = len(train_tokens) * (chars_per_superbpe / gpt2_chars_per_token)
    reduction = (1 - len(train_tokens) / estimated_gpt2_tokens) * 100
    
    print(f"SuperBPE chars/token: {chars_per_superbpe:.2f}")
    print(f"GPT-2 chars/token (typical): {gpt2_chars_per_token:.2f}")
    print(f"Estimated GPT-2 tokens: {int(estimated_gpt2_tokens):,}")
    print(f"SuperBPE tokens: {len(train_tokens):,}")
    print(f"Estimated reduction: {reduction:.1f}%")
    print(f"Speedup factor: {estimated_gpt2_tokens/len(train_tokens):.2f}x")
    
    print("\n" + "="*60)
    print("DATASET READY!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Total tokens: {len(train_tokens) + len(val_tokens):,}")
    print("\nTo use this dataset:")
    print("  python train_tinystories.py --data_dir data_textbooks_superbpe \\")
    print("    --vocab_size 200005 --factorized --embedding_rank 128 \\")
    print("    --max_iters 1000")
    print("\nExpected benefits:")
    print("  - Higher quality educational content")
    print("  - Better test of model comprehension")
    print("  - ~40% faster training with SuperBPE")

if __name__ == "__main__":
    prepare_dataset()