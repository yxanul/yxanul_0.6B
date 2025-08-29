#!/usr/bin/env python3
"""
Prepare experimental-pretrain-1b dataset using SuperBPE tokenizer.
Creates train.bin and val.bin with proper EOS tokens between documents.
Uses the 'text' column from the parquet file.
Optimized for multi-core CPU processing of large datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def tokenize_batch(texts, tokenizer_path):
    """Tokenize a batch of texts using a fresh tokenizer instance."""
    # Load tokenizer in each worker process
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=True,
        local_files_only=True
    )
    
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    
    all_tokens = []
    for text in texts:
        if pd.isna(text) or len(text.strip()) == 0:
            continue
        
        # Tokenize the document
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Add tokens and EOS separator
        all_tokens.extend(tokens)
        all_tokens.append(eos_token_id)
    
    return all_tokens

def prepare_dataset():
    """Prepare experimental-pretrain-1b with SuperBPE tokenizer."""
    
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
    
    # Look for experimental-pretrain-1b dataset
    data_dir = None
    possible_locations = [
        Path('experimental-pretrain-1b'),  # In experimental directory
        Path('../experimental-pretrain-1b'),  # Parent directory
        Path('/workspace/yxanul_0.6B/experimental-pretrain-1b'),  # Absolute path on RunPod
    ]
    
    for location in possible_locations:
        if location.exists():
            # Check if it has the expected parquet file
            train_file = location / 'dataset_1b.parquet'
            if train_file.exists():
                data_dir = location
                print(f"Found experimental-pretrain-1b dataset at: {data_dir}")
                break
    
    if data_dir is None:
        print("\nERROR: Cannot find experimental-pretrain-1b dataset!")
        print("Please ensure dataset_1b.parquet is in one of these locations:")
        for loc in possible_locations:
            print(f"     - {loc}/dataset_1b.parquet")
        raise FileNotFoundError("experimental-pretrain-1b dataset not found - see instructions above")
    
    output_dir = Path('data_experimental_1b_superbpe')
    output_dir.mkdir(exist_ok=True)
    
    # Process training data
    print("\nProcessing training data...")
    train_file = data_dir / 'dataset_1b.parquet'
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    print(f"Loading {train_file}...")
    print("Note: This is a large file, loading may take a moment...")
    df = pd.read_parquet(train_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Use the 'text' column as specified
    if 'text' not in df.columns:
        raise ValueError(f"'text' column not found! Available columns: {df.columns.tolist()}")
    
    texts = df['text'].tolist()
    print(f"Processing {len(texts):,} mixed documents (educational, math, code)...")
    
    # Determine number of CPU cores to use
    num_cores = mp.cpu_count()
    # Use 75% of available cores to leave some for system
    num_workers = max(1, int(num_cores * 0.75))
    print(f"Using {num_workers} CPU cores for parallel tokenization (out of {num_cores} available)")
    
    # Tokenize each document and add EOS token between them
    all_train_tokens = []
    
    # Process in chunks for multiprocessing
    chunk_size = 1000  # Documents per chunk
    num_chunks = (len(texts) + chunk_size - 1) // chunk_size
    
    print(f"Processing in {num_chunks} chunks of up to {chunk_size} documents each...")
    
    # Create a partial function with the tokenizer path
    tokenize_func = partial(tokenize_batch, tokenizer_path=t80k_path)
    
    # Process with multiprocessing pool
    with mp.Pool(num_workers) as pool:
        for i in tqdm(range(0, len(texts), chunk_size), desc="Tokenizing documents", total=num_chunks):
            batch = texts[i:min(i+chunk_size, len(texts))]
            
            # Split batch across workers
            worker_batch_size = max(1, len(batch) // num_workers)
            worker_batches = [batch[j:j+worker_batch_size] 
                            for j in range(0, len(batch), worker_batch_size)]
            
            # Process in parallel
            results = pool.map(tokenize_func, worker_batches)
            
            # Combine results
            for tokens in results:
                all_train_tokens.extend(tokens)
    
    train_tokens = np.array(all_train_tokens, dtype=np.uint32)  # uint32 for vocab > 65k
    print(f"Total train tokens: {len(train_tokens):,} from {len(texts):,} documents")
    
    # Calculate average tokens per document
    avg_tokens = len(train_tokens) / len(texts)
    print(f"Average tokens per document: {avg_tokens:.1f}")
    
    # Split into train/val (95/5 split for large dataset)
    split_idx = int(len(train_tokens) * 0.95)
    val_tokens = train_tokens[split_idx:]
    train_tokens = train_tokens[:split_idx]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Save training data
    train_output = output_dir / 'train.bin'
    # CRITICAL: Use uint32 for SuperBPE (vocab=200k > uint16 max=65k)
    train_tokens.astype(np.uint32).tofile(train_output)
    print(f"Saved training data to {train_output}")
    
    # Save validation data
    val_output = output_dir / 'val.bin'
    # CRITICAL: Use uint32 for SuperBPE (vocab=200k > uint16 max=65k)
    val_tokens.astype(np.uint32).tofile(val_output)
    print(f"Saved validation data to {val_output}")
    
    # Save tokenizer config for reference
    config = {
        'dataset': 'experimental-pretrain-1b',
        'tokenizer_name': 'superbpe-t80k',
        'vocab_size': len(tokenizer),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'total_documents': len(texts),
        'avg_tokens_per_doc': avg_tokens,
        'eos_token_id': eos_token_id,
        'description': 'Mixed high-quality dataset: educational content, mathematical reasoning, Python code',
        'num_workers_used': num_workers,
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
    
    # Calculate total size in GB
    total_size_gb = (len(train_tokens) + len(val_tokens)) * 4 / 1e9  # 4 bytes per uint32
    
    print("\n" + "="*60)
    print("DATASET READY!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Total tokens: {len(train_tokens) + len(val_tokens):,}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print("\nTo use this dataset:")
    print("  python train_tinystories.py --data_dir data_experimental_1b_superbpe \\")
    print("    --vocab_size 200005 --factorized --embedding_rank 128 \\")
    print("    --learning_rate 3e-4 --max_iters 2500 --eval_interval 100")
    print("\nExpected benefits over tiny-textbooks:")
    print("  - Larger scale training (1B tokens)")
    print("  - Mixed content: educational, mathematical, code")
    print("  - Better generalization across domains")
    print("  - ~40% faster training with SuperBPE")
    print("\nRecommended hardware:")
    print("  - A100 40GB or 80GB for optimal batch sizes")
    print("  - H100 for maximum training speed")
    print(f"  - RTX 4090 can handle with gradient accumulation")

if __name__ == "__main__":
    prepare_dataset()