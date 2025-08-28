#!/usr/bin/env python3
"""
Prepare TinyStories dataset - Fixed version.
Either use downgraded datasets or clone the repo directly.
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    # Configuration
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    print("TinyStories Dataset Preparation")
    print("=" * 60)
    
    # Load tokenizer
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        print("Installing tiktoken...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tiktoken"])
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    
    # Method 1: Try with current datasets version
    try:
        from datasets import load_dataset
        print("Attempting to load with current datasets version...")
        dataset = load_dataset("roneneldan/TinyStories")
        print("Success! Dataset loaded.")
        
    except Exception as e:
        print(f"Current version failed: {e}")
        print("\nChoose a fix method:")
        print("1. Downgrade datasets to 2.14.0 (recommended)")
        print("2. Clone the repository and load manually")
        print("3. Exit")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            # Downgrade and retry
            print("\nDowngrading datasets library...")
            subprocess.run([sys.executable, "-m", "pip", "install", "datasets==2.14.0"], check=True)
            
            # Reload the module
            import importlib
            import datasets
            importlib.reload(datasets)
            from datasets import load_dataset
            
            print("Retrying with datasets 2.14.0...")
            dataset = load_dataset("roneneldan/TinyStories")
            print("Success! Dataset loaded with older version.")
            
        elif choice == "2":
            # Clone repo method
            print("\nCloning TinyStories repository...")
            repo_dir = Path("TinyStories-repo")
            
            if not repo_dir.exists():
                subprocess.run([
                    "git", "clone", 
                    "https://huggingface.co/datasets/roneneldan/TinyStories",
                    str(repo_dir)
                ], check=True)
            
            # Look for data files
            print(f"Looking for data files in {repo_dir}...")
            data_files = list(repo_dir.glob("*.txt")) + list(repo_dir.glob("*.csv")) + list(repo_dir.glob("*.parquet"))
            
            if not data_files:
                print("No data files found. Checking subdirectories...")
                data_files = list(repo_dir.rglob("*.txt")) + list(repo_dir.rglob("*.csv")) + list(repo_dir.rglob("*.parquet"))
            
            print(f"Found files: {[f.name for f in data_files]}")
            
            # Load the data manually
            dataset = {'train': [], 'validation': []}
            
            for file in data_files:
                if 'train' in file.name.lower():
                    split = 'train'
                elif 'valid' in file.name.lower() or 'val' in file.name.lower():
                    split = 'validation'
                else:
                    continue
                
                print(f"Loading {file.name} into {split} split...")
                
                if file.suffix == '.txt':
                    with open(file, 'r', encoding='utf-8') as f:
                        text = f.read()
                        stories = [s.strip() for s in text.split('\n\n') if s.strip()]
                        dataset[split].extend([{'text': s} for s in stories])
                        
                elif file.suffix == '.csv':
                    import pandas as pd
                    df = pd.read_csv(file)
                    # Assume the text column is named 'text' or 'story'
                    text_col = 'text' if 'text' in df.columns else 'story'
                    dataset[split].extend(df[text_col].apply(lambda x: {'text': x}).tolist())
                    
                elif file.suffix == '.parquet':
                    import pandas as pd
                    df = pd.read_parquet(file)
                    text_col = 'text' if 'text' in df.columns else 'story'
                    dataset[split].extend(df[text_col].apply(lambda x: {'text': x}).tolist())
            
            print(f"Loaded {len(dataset['train'])} train examples")
            print(f"Loaded {len(dataset['validation'])} validation examples")
            
        else:
            print("Exiting...")
            sys.exit(0)
    
    # Process each split
    for split in ['train', 'validation']:
        if split not in dataset or not dataset[split]:
            print(f"Warning: {split} split is empty, skipping...")
            continue
            
        print(f"\nProcessing {split} split...")
        
        # Handle different dataset formats
        if hasattr(dataset[split], '__iter__'):
            data = dataset[split]
        else:
            data = dataset[split]
        
        # Tokenize all texts
        print(f"Tokenizing {len(data)} examples...")
        all_tokens = []
        
        for example in tqdm(data, desc=f"Tokenizing {split}"):
            # Extract text from various formats
            if isinstance(example, dict):
                text = example.get('text', example.get('story', ''))
            elif hasattr(example, 'get'):
                text = example.get('text', example.get('story', ''))
            elif hasattr(example, '__getitem__'):
                text = example['text'] if 'text' in example else example['story']
            else:
                text = str(example)
            
            if text:  # Skip empty texts
                tokens = enc.encode_ordinary(text)
                all_tokens.extend(tokens)
                # Add EOS token
                all_tokens.append(enc.eot_token)
        
        # Convert to numpy array
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        
        # Save as memory-mapped file
        output_file = output_dir / f"{split}.bin"
        print(f"Writing {len(all_tokens):,} tokens to {output_file}")
        
        # Create memory-mapped file
        arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(len(all_tokens),))
        arr[:] = all_tokens
        arr.flush()
        
        print(f"{split}: {len(all_tokens):,} tokens saved")
    
    # Save metadata
    train_file = output_dir / 'train.bin'
    val_file = output_dir / 'validation.bin'
    
    if train_file.exists() and val_file.exists():
        metadata = {
            'vocab_size': enc.n_vocab,
            'train_tokens': len(np.memmap(train_file, dtype=np.uint16, mode='r')),
            'val_tokens': len(np.memmap(val_file, dtype=np.uint16, mode='r')),
        }
        
        print("\n" + "=" * 60)
        print("Dataset preparation complete!")
        print(f"Train tokens: {metadata['train_tokens']:,}")
        print(f"Validation tokens: {metadata['val_tokens']:,}")
        print(f"Total tokens: {metadata['train_tokens'] + metadata['val_tokens']:,}")
        print("=" * 60)
    else:
        print("\nWarning: Some files may be missing")
        if train_file.exists():
            print(f"train.bin exists: {train_file.stat().st_size / 1024 / 1024:.1f} MB")
        if val_file.exists():
            print(f"validation.bin exists: {val_file.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()