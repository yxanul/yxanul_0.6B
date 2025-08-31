#!/usr/bin/env python3
"""
Prepare mixed-domain dataset for training.
Processes half of the 3B token mixed dataset (~1.5B tokens).
Dataset includes diverse content: text, code, math, etc.
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
    Prepare mixed-domain dataset (half of 3B tokens).
    More diverse than FineWeb-Edu - includes code, math, text.
    """
    
    print("="*60)
    print("MIXED-DOMAIN DATASET PREPARATION")
    print("Processing 1.5B tokens from mixed-pretrain-3b")
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
    print("Processing HALF of the 3B tokens (~1.5B tokens)")
    
    # Create output directory
    output_dir = Path('data_mixed_3b')
    output_dir.mkdir(exist_ok=True)
    
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
    
    # Process HALF of the dataset for quick testing
    half_docs = total_docs // 2
    print(f"Processing first {half_docs:,} documents (50% of dataset)")
    df = df.head(half_docs)
    
    texts = df['text'].tolist()
    total_chars = sum(len(text) for text in texts if pd.notna(text))
    
    # Process in batches
    print("\nTokenizing documents...")
    all_train_tokens = []
    batch_size = 5000
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:min(i+batch_size, len(texts))]
        tokens = tokenize_batch_fast(batch, tokenizer, eos_token_id)
        all_train_tokens.extend(tokens)
        
        if i % (batch_size * 10) == 0 and i > 0:
            print(f"  Processed {i:,} docs, current tokens: {len(all_train_tokens):,}")
    
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
        'dataset': 'mixed-pretrain-3b',
        'mode': 'half_dataset',
        'tokenizer': 'HuggingFaceTB/SmolLM2-135M',
        'vocab_size': len(tokenizer),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'total_documents': half_docs,
        'dataset_type': 'mixed-domain (text, code, math)',
    }
    
    with open(output_dir / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("MIXED DATASET READY!")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Size: {len(train_tokens)/1e9:.1f}B tokens")
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