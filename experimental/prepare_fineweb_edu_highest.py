#!/usr/bin/env python3
"""
Prepare highest-quality FineWeb-Edu dataset using SuperBPE tokenizer.
Uses pre-filtered dataset with educational score >= 3.5, language score >= 0.95.
Processes ~1/3 of the 4.17B tokens for manageable training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import platform

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

def prepare_dataset(max_files=None):
    """
    Prepare FineWeb-Edu highest quality dataset with SuperBPE tokenizer.
    
    Args:
        max_files: Number of parquet files to process (None = all files)
                   Use ~1/3 of files for 1.4B tokens
    """
    
    print("="*60)
    print("FINEWEB-EDU HIGHEST QUALITY DATASET PREPARATION")
    print("="*60)
    
    # Load SuperBPE tokenizer from local cache
    print("\nLoading SuperBPE tokenizer from local cache...")
    
    # Determine cache path based on platform
    if platform.system() == "Linux":
        # RunPod instance paths
        cache_base = Path("/workspace/yxanul_0.6B/tokenizer_cache")
        data_base = Path("/workspace/yxanul_0.6B/experimental/fineweb-edu-highest-quality-2025/data")
    else:
        # Local Windows paths
        cache_base = Path("D:/ai_testing/yxanul_0.6B/tokenizer_cache")
        data_base = Path("D:/ai_testing/yxanul_0.6B/experimental/fineweb-edu-highest-quality-2025/data")
    
    # Check if cache exists
    if not cache_base.exists():
        cache_base = Path("../tokenizer_cache")
        if not cache_base.exists():
            raise FileNotFoundError(f"Tokenizer cache not found at {cache_base}")
    
    # Load SuperBPE-t80k tokenizer
    t80k_path = cache_base / "superbpe-t80k-fast"
    if not t80k_path.exists():
        raise FileNotFoundError(f"SuperBPE-t80k cache not found at {t80k_path}")
    
    print(f"Found t80k cache at {t80k_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(t80k_path),
            use_fast=True,  # Fast Rust tokenizer
            local_files_only=True
        )
        print(f"SUCCESS: Loaded SuperBPE-t80k fast tokenizer!")
    except Exception as e:
        raise RuntimeError(f"Failed to load SuperBPE-t80k tokenizer: {e}")
    
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Get EOS token ID for document separation
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    print(f"EOS token ID: {eos_token_id}")
    
    # Check for dataset directory
    if not data_base.exists():
        # Try relative path
        data_base = Path("fineweb-edu-highest-quality-2025/data")
        if not data_base.exists():
            print(f"\nERROR: Cannot find dataset at {data_base}")
            print("Please ensure fineweb-edu-highest-quality-2025/data/ exists with parquet files")
            raise FileNotFoundError("Dataset not found")
    
    # Get all parquet files
    parquet_files = sorted(data_base.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_base}")
    
    print(f"\nFound {len(parquet_files)} parquet files")
    
    # Limit files if specified (for 1/3 of dataset)
    if max_files:
        parquet_files = parquet_files[:max_files]
        print(f"Processing first {max_files} files (~{max_files/len(parquet_files)*100:.0f}% of dataset)")
    else:
        # Default to 1/3 of files for ~1.4B tokens
        third_files = len(parquet_files) // 3
        parquet_files = parquet_files[:third_files]
        print(f"Processing {third_files} files (1/3 of dataset, ~1.4B tokens)")
    
    # Create output directory
    output_dir = Path('data_fineweb_edu_highest_superbpe')
    output_dir.mkdir(exist_ok=True)
    
    # Process all selected files
    print(f"\nProcessing {len(parquet_files)} parquet files...")
    all_train_tokens = []
    total_docs = 0
    total_chars = 0
    
    # Process each parquet file
    for file_idx, parquet_file in enumerate(tqdm(parquet_files, desc="Processing files")):
        print(f"\n  Loading {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)
        
        # Check columns
        if 'text' not in df.columns:
            print(f"  Warning: 'text' column not found in {parquet_file.name}")
            print(f"  Available columns: {df.columns.tolist()}")
            continue
        
        texts = df['text'].tolist()
        print(f"  Documents in file: {len(texts):,}")
        
        # Track statistics
        total_docs += len(texts)
        total_chars += sum(len(text) for text in texts if pd.notna(text))
        
        # Process in batches for speed
        batch_size = 5000  # Large batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:min(i+batch_size, len(texts))]
            tokens = tokenize_batch_fast(batch, tokenizer, eos_token_id)
            all_train_tokens.extend(tokens)
        
        print(f"  Running total: {len(all_train_tokens):,} tokens from {total_docs:,} documents")
    
    # Convert to numpy array
    print("\nConverting to numpy array...")
    train_tokens = np.array(all_train_tokens, dtype=np.uint32)
    
    print(f"\nDataset statistics:")
    print(f"  Documents processed: {total_docs:,}")
    print(f"  Total tokens: {len(train_tokens):,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Avg tokens per doc: {len(train_tokens)/total_docs:.1f}")
    print(f"  Avg chars per token: {total_chars/len(train_tokens):.2f}")
    
    # Estimate token reduction vs GPT-2
    chars_per_superbpe = total_chars / len(train_tokens)
    gpt2_chars_per_token = 4.3  # Typical for English
    estimated_gpt2_tokens = len(train_tokens) * (chars_per_superbpe / gpt2_chars_per_token)
    reduction = (1 - len(train_tokens) / estimated_gpt2_tokens) * 100
    
    print(f"\nTokenization efficiency:")
    print(f"  SuperBPE chars/token: {chars_per_superbpe:.2f}")
    print(f"  GPT-2 chars/token (typical): {gpt2_chars_per_token:.2f}")
    print(f"  Estimated GPT-2 tokens: {int(estimated_gpt2_tokens):,}")
    print(f"  SuperBPE tokens: {len(train_tokens):,}")
    print(f"  Token reduction: {reduction:.1f}%")
    print(f"  Speedup factor: {estimated_gpt2_tokens/len(train_tokens):.2f}x")
    
    # Split into train/val (95/5 for large dataset)
    split_idx = int(len(train_tokens) * 0.95)
    val_tokens = train_tokens[split_idx:]
    train_tokens = train_tokens[:split_idx]
    
    print(f"\nFinal split:")
    print(f"  Train tokens: {len(train_tokens):,}")
    print(f"  Val tokens: {len(val_tokens):,}")
    
    # Save training data
    train_output = output_dir / 'train.bin'
    train_tokens.astype(np.uint32).tofile(train_output)
    print(f"\nSaved training data to {train_output}")
    
    # Save validation data
    val_output = output_dir / 'val.bin'
    val_tokens.astype(np.uint32).tofile(val_output)
    print(f"Saved validation data to {val_output}")
    
    # Save dataset config
    config = {
        'dataset': 'fineweb-edu-highest-quality-2025',
        'quality_criteria': {
            'min_tokens': 1000,
            'min_edu_score': 3.5,
            'min_lang_score': 0.95
        },
        'tokenizer': 'superbpe-t80k',
        'vocab_size': len(tokenizer),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'total_documents': total_docs,
        'files_processed': len(parquet_files),
        'avg_tokens_per_doc': len(train_tokens) / total_docs,
        'chars_per_token': chars_per_superbpe,
        'token_reduction_vs_gpt2': f"{reduction:.1f}%",
        'eos_token_id': eos_token_id,
    }
    
    with open(output_dir / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Calculate size
    total_size_gb = (len(train_tokens) + len(val_tokens)) * 4 / 1e9
    
    print("\n" + "="*60)
    print("DATASET READY!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"Quality: Top 2% of FineWeb-Edu (score >= 3.5)")
    print("\nTo train:")
    print("  python train_tinystories.py --data_dir data_fineweb_edu_highest_superbpe \\")
    print("    --vocab_size 200005 --factorized --embedding_rank 128 \\")
    print("    --learning_rate 3e-4 --max_iters 5000")
    print("\nExpected benefits over mixed dataset:")
    print("  - Pure high-quality English text")
    print("  - No domain confusion (math/code mixing)")
    print("  - Factually accurate educational content")
    print("  - Coherent, GPT-2-like base model")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_files', type=int, default=None,
                       help='Number of parquet files to process (None = 1/3 of total)')
    args = parser.parse_args()
    
    prepare_dataset(args.max_files)