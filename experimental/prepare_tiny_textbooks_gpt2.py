#!/usr/bin/env python3
"""
Prepare tiny-textbooks dataset using GPT-2 tokenizer.
Fallback option if SuperBPE causes issues with sparse vocabulary.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tiktoken
import json
from tqdm import tqdm

def prepare_dataset():
    """Prepare tiny-textbooks with GPT-2 tokenizer."""
    
    print("Loading GPT-2 tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print(f"Vocabulary size: {tokenizer.n_vocab}")
    
    # Get EOS token ID
    eos_token_id = tokenizer.eot_token
    print(f"EOS token ID: {eos_token_id}")
    
    # Look for tiny-textbooks dataset
    data_dir = None
    possible_locations = [
        Path('tiny-textbooks/tiny-textbooks'),
        Path('tiny-textbooks'),
        Path('../tiny-textbooks/tiny-textbooks'),
        Path('/workspace/tiny-textbooks/tiny-textbooks'),
    ]
    
    for location in possible_locations:
        if location.exists():
            train_file = location / 'train-00000-of-00001.parquet'
            if train_file.exists():
                data_dir = location
                print(f"Found tiny-textbooks dataset at: {data_dir}")
                break
    
    if data_dir is None:
        print("\nERROR: Cannot find tiny-textbooks dataset!")
        raise FileNotFoundError("tiny-textbooks dataset not found")
    
    output_dir = Path('data_textbooks_gpt2')
    output_dir.mkdir(exist_ok=True)
    
    # Process training data
    print("\nProcessing training data...")
    train_file = data_dir / 'train-00000-of-00001.parquet'
    
    print(f"Loading {train_file}...")
    df = pd.read_parquet(train_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Using 'textbook' column")
    
    if 'textbook' not in df.columns:
        raise ValueError(f"'textbook' column not found! Available: {df.columns.tolist()}")
    
    texts = df['textbook'].tolist()
    print(f"Processing {len(texts):,} textbook documents...")
    
    # Tokenize with GPT-2
    all_train_tokens = []
    
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing with GPT-2"):
        batch = texts[i:min(i+batch_size, len(texts))]
        
        for text in batch:
            if pd.isna(text) or len(text.strip()) == 0:
                continue
            
            # GPT-2 tokenization
            tokens = tokenizer.encode_ordinary(text)
            
            # Add tokens and EOS separator
            all_train_tokens.extend(tokens)
            all_train_tokens.append(eos_token_id)
    
    train_tokens = np.array(all_train_tokens, dtype=np.uint32)
    print(f"Total GPT-2 tokens: {len(train_tokens):,} from {len(texts):,} documents")
    
    # Calculate statistics
    avg_tokens = len(train_tokens) / len(texts)
    print(f"Average tokens per document: {avg_tokens:.1f}")
    
    # Compare with SuperBPE if available
    superbpe_path = Path('data_textbooks_superbpe/dataset_config.json')
    if superbpe_path.exists():
        with open(superbpe_path, 'r') as f:
            superbpe_config = json.load(f)
        superbpe_tokens = superbpe_config['train_tokens'] + superbpe_config['val_tokens']
        reduction = (1 - superbpe_tokens / len(train_tokens)) * 100
        print(f"\nComparison with SuperBPE:")
        print(f"  GPT-2 tokens: {len(train_tokens):,}")
        print(f"  SuperBPE tokens: {superbpe_tokens:,}")
        print(f"  SuperBPE reduction: {reduction:.1f}%")
    
    # Split into train/val (90/10)
    split_idx = int(len(train_tokens) * 0.9)
    val_tokens = train_tokens[split_idx:]
    train_tokens = train_tokens[:split_idx]
    
    print(f"\nFinal split:")
    print(f"  Train tokens: {len(train_tokens):,}")
    print(f"  Val tokens: {len(val_tokens):,}")
    
    # Calculate tokens per vocabulary entry
    tokens_per_vocab = len(train_tokens) / tokenizer.n_vocab
    print(f"  Tokens per vocab entry: {tokens_per_vocab:.0f} (higher is better)")
    
    # Save training data
    train_output = output_dir / 'train.bin'
    train_tokens.astype(np.uint16).tofile(train_output)
    print(f"\nSaved training data to {train_output}")
    
    # Save validation data
    val_output = output_dir / 'val.bin'
    val_tokens.astype(np.uint16).tofile(val_output)
    print(f"Saved validation data to {val_output}")
    
    # Save config
    config = {
        'dataset': 'tiny-textbooks',
        'tokenizer': 'gpt2',
        'vocab_size': tokenizer.n_vocab,
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'total_documents': len(texts),
        'avg_tokens_per_doc': avg_tokens,
        'tokens_per_vocab_entry': tokens_per_vocab,
        'eos_token_id': eos_token_id,
    }
    
    with open(output_dir / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("DATASET READY!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Vocabulary size: {tokenizer.n_vocab}")
    print(f"Total tokens: {len(train_tokens) + len(val_tokens):,}")
    print("\nTo train with GPT-2 tokenizer:")
    print("  python train_tinystories.py --data_dir data_textbooks_gpt2 \\")
    print("    --vocab_size 50257 --factorized --embedding_rank 128 \\")
    print("    --learning_rate 3e-4 --max_iters 2000")
    print("\nExpected benefits over SuperBPE:")
    print("  - More stable training (4x more examples per token)")
    print("  - Better generation quality")
    print("  - No sparse vocabulary issues")
    print("  - Proven tokenizer for English text")

if __name__ == "__main__":
    prepare_dataset()