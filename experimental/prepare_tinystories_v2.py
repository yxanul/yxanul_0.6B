#!/usr/bin/env python3
"""
Prepare TinyStories dataset for training - Version 2.
Handles datasets library compatibility issues.
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    # Try different import strategies
    try:
        from datasets import load_dataset
        import tiktoken
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Install with: pip install datasets tiktoken")
        sys.exit(1)
    
    # Configuration
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Load tokenizer (GPT-2 tokenizer for compatibility)
    print("Loading tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    
    # Load TinyStories dataset with multiple fallback methods
    print("Loading TinyStories dataset...")
    dataset = None
    
    # Method 1: Try standard loading
    try:
        print("Attempting standard load...")
        dataset = load_dataset("roneneldan/TinyStories")
        print("Successfully loaded with standard method")
    except Exception as e:
        print(f"Standard load failed: {e}")
    
    # Method 2: Try with trust_remote_code
    if dataset is None:
        try:
            print("Attempting with trust_remote_code=True...")
            dataset = load_dataset("roneneldan/TinyStories", trust_remote_code=True)
            print("Successfully loaded with trust_remote_code")
        except Exception as e:
            print(f"trust_remote_code method failed: {e}")
    
    # Method 3: Try direct download and manual processing
    if dataset is None:
        try:
            print("Attempting direct download...")
            # Download specific splits
            from huggingface_hub import hf_hub_download, list_repo_files
            
            repo_id = "roneneldan/TinyStories"
            files = list_repo_files(repo_id)
            print(f"Found files: {files}")
            
            # Look for parquet files
            parquet_files = [f for f in files if f.endswith('.parquet')]
            if parquet_files:
                import pandas as pd
                
                # Create dataset dict manually
                dataset = {}
                for file in parquet_files:
                    print(f"Downloading {file}...")
                    local_path = hf_hub_download(repo_id=repo_id, filename=file)
                    
                    # Read parquet file
                    df = pd.read_parquet(local_path)
                    
                    # Determine split from filename
                    if 'train' in file.lower():
                        dataset['train'] = df
                    elif 'valid' in file.lower() or 'val' in file.lower():
                        dataset['validation'] = df
                
                print("Successfully loaded via direct download")
            else:
                raise ValueError("No parquet files found in repository")
                
        except Exception as e:
            print(f"Direct download failed: {e}")
            print("\nFailed to load dataset. Please try:")
            print("1. pip install --upgrade datasets")
            print("2. pip install huggingface_hub pandas pyarrow")
            sys.exit(1)
    
    # Process each split
    for split in ['train', 'validation']:
        print(f"\nProcessing {split} split...")
        
        # Get the data
        if isinstance(dataset, dict) and split in dataset:
            # If we loaded manually
            data = dataset[split]
            if hasattr(data, 'to_dict'):
                # It's a DataFrame
                data = data.to_dict('records')
        else:
            # Standard datasets object
            data = dataset[split]
        
        # Tokenize all texts
        print(f"Tokenizing {len(data)} examples...")
        all_tokens = []
        
        for example in tqdm(data, desc=f"Tokenizing {split}"):
            # Handle different data formats
            if isinstance(example, dict):
                text = example.get('text', example.get('story', ''))
            else:
                text = example['text']
            
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
    metadata = {
        'vocab_size': enc.n_vocab,
        'train_tokens': len(np.memmap(output_dir / 'train.bin', dtype=np.uint16, mode='r')),
        'val_tokens': len(np.memmap(output_dir / 'validation.bin', dtype=np.uint16, mode='r')),
    }
    
    print("\nDataset preparation complete!")
    print(f"Train tokens: {metadata['train_tokens']:,}")
    print(f"Validation tokens: {metadata['val_tokens']:,}")
    print(f"Total tokens: {metadata['train_tokens'] + metadata['val_tokens']:,}")

if __name__ == "__main__":
    main()