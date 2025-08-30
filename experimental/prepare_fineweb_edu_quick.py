#!/usr/bin/env python3
"""
Quick version: Prepare 1/4 of FineWeb-Edu dataset for fast testing.
Processes ~500M tokens in ~10 minutes for rapid experimentation.
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

def prepare_dataset_quick():
    """
    Quick version: Prepare 1/4 of FineWeb-Edu for testing.
    Processes ~500M tokens in about 10 minutes.
    """
    
    print("="*60)
    print("QUICK DATASET PREPARATION (1/4 size for testing)")
    print("With SmolLM2-135M Tokenizer")
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
        data_base = Path("/workspace/yxanul_0.6B/experimental/fineweb-edu-highest-quality-2025/data")
    else:
        # Local Windows paths
        data_base = Path("D:/ai_testing/yxanul_0.6B/experimental/fineweb-edu-highest-quality-2025/data")
    
    # Check for dataset directory
    if not data_base.exists():
        # Try relative path
        data_base = Path("fineweb-edu-highest-quality-2025/data")
        if not data_base.exists():
            print(f"\nERROR: Cannot find dataset at {data_base}")
            print("Please ensure fineweb-edu-highest-quality-2025/data/ exists")
            raise FileNotFoundError("Dataset not found")
    
    # Get all parquet files
    parquet_files = sorted(data_base.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_base}")
    
    total_files = len(parquet_files)
    print(f"\nFound {total_files} parquet files total")
    
    # QUICK MODE: Process only 1/4 of files
    quarter_files = max(1, total_files // 4)  # At least 1 file
    parquet_files = parquet_files[:quarter_files]
    print(f"QUICK MODE: Processing only {quarter_files} files (1/4 of dataset)")
    print(f"Expected: ~500M tokens, ~10 minutes processing time")
    
    # Create output directory
    output_dir = Path('data_fineweb_edu_quick')
    output_dir.mkdir(exist_ok=True)
    
    # Process files
    print(f"\nProcessing {len(parquet_files)} parquet files...")
    all_train_tokens = []
    total_docs = 0
    total_chars = 0
    
    for file_idx, parquet_file in enumerate(tqdm(parquet_files, desc="Processing files")):
        print(f"\n  [{file_idx+1}/{len(parquet_files)}] Loading {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)
        
        if 'text' not in df.columns:
            print(f"  Warning: 'text' column not found, skipping")
            continue
        
        texts = df['text'].tolist()
        print(f"  Documents: {len(texts):,}")
        
        # Track statistics
        total_docs += len(texts)
        total_chars += sum(len(text) for text in texts if pd.notna(text))
        
        # Process in batches
        batch_size = 5000
        for i in range(0, len(texts), batch_size):
            batch = texts[i:min(i+batch_size, len(texts))]
            tokens = tokenize_batch_fast(batch, tokenizer, eos_token_id)
            all_train_tokens.extend(tokens)
        
        print(f"  Running total: {len(all_train_tokens):,} tokens")
    
    # Convert to numpy array
    print("\nConverting to numpy array...")
    train_tokens = np.array(all_train_tokens, dtype=np.uint16)
    
    print(f"\nDataset statistics:")
    print(f"  Documents: {total_docs:,}")
    print(f"  Total tokens: {len(train_tokens):,}")
    print(f"  Total characters: {total_chars:,}")
    if total_docs > 0:
        print(f"  Avg tokens/doc: {len(train_tokens)/total_docs:.1f}")
        print(f"  Avg chars/token: {total_chars/len(train_tokens):.2f}")
    
    # Split into train/val (95/5)
    split_idx = int(len(train_tokens) * 0.95)
    val_tokens = train_tokens[split_idx:]
    train_tokens = train_tokens[:split_idx]
    
    print(f"\nFinal split:")
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val: {len(val_tokens):,} tokens")
    
    # Save files
    train_output = output_dir / 'train.bin'
    train_tokens.astype(np.uint16).tofile(train_output)
    print(f"\nSaved training data to {train_output}")
    
    val_output = output_dir / 'val.bin'
    val_tokens.astype(np.uint16).tofile(val_output)
    print(f"Saved validation data to {val_output}")
    
    # Save config
    config = {
        'dataset': 'fineweb-edu-highest-quality-2025-QUICK',
        'mode': 'quick_test',
        'tokenizer': 'HuggingFaceTB/SmolLM2-135M',
        'vocab_size': len(tokenizer),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'total_documents': total_docs,
        'files_processed': len(parquet_files),
    }
    
    with open(output_dir / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("QUICK DATASET READY!")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Size: ~{len(train_tokens)/1e6:.0f}M tokens")
    print("\nTo test training quickly:")
    print("  # Test if FP8 works on your hardware:")
    print("  python model_te.py")
    print("\n  # If FP8 works (H100 only):")
    print("  python train_fp8.py --data_dir data_fineweb_edu_quick \\")
    print("    --max_iters 100 --eval_interval 20")
    print("\n  # If FP8 doesn't work (A100/4090/5090):")
    print("  python train_tinystories.py --data_dir data_fineweb_edu_quick \\")
    print("    --vocab_size 49152 --max_iters 100 --eval_interval 20")

if __name__ == "__main__":
    prepare_dataset_quick()