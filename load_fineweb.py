#!/usr/bin/env python3
"""
Load and clean FineWeb-Edu dataset from parquet files.
Handles schema issues with __index_level_0__ column.
"""

import os
import sys
import glob
import pandas as pd
from datasets import Dataset
from pathlib import Path
import gc

def load_fineweb_dataset(max_files=None):
    """Load FineWeb-Edu dataset from local parquet files."""
    
    print("=" * 60)
    print("FineWeb-Edu Dataset Loader")
    print("=" * 60)
    
    # Check multiple possible locations
    possible_paths = [
        "/workspace/fineweb-edu-highest-quality-2025/data",
        "/workspace/yxanul_0.6B/fineweb-edu-highest-quality-2025/data",
        "./fineweb-edu-highest-quality-2025/data",
        "./data/fineweb-edu",
    ]
    
    parquet_files = []
    dataset_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            pattern = os.path.join(path, "*.parquet")
            files = glob.glob(pattern)
            if files:
                parquet_files = sorted(files)
                dataset_path = path
                break
    
    if not parquet_files:
        print("ERROR: No parquet files found!")
        print("Please clone the dataset first:")
        print("  cd /workspace")
        print("  git clone https://huggingface.co/datasets/Yxanul/fineweb-edu-highest-quality-2025")
        print("  cd fineweb-edu-highest-quality-2025")
        print("  git lfs pull")
        sys.exit(1)
    
    print(f"Found {len(parquet_files)} parquet files in {dataset_path}")
    
    # Limit files if specified
    if max_files:
        parquet_files = parquet_files[:max_files]
        print(f"Loading first {max_files} files for testing")
    
    # Load and clean data
    all_data = []
    total_examples = 0
    
    for i, file_path in enumerate(parquet_files):
        try:
            print(f"\nLoading file {i+1}/{len(parquet_files)}: {os.path.basename(file_path)}")
            
            # Read parquet file
            df = pd.read_parquet(file_path)
            initial_len = len(df)
            
            # Remove problematic columns
            cols_to_drop = ['__index_level_0__']
            for col in cols_to_drop:
                if col in df.columns:
                    df = df.drop(col, axis=1)
                    print(f"  Dropped column: {col}")
            
            # Keep only necessary columns for training
            if 'text' in df.columns:
                # Create dict for each example
                for _, row in df.iterrows():
                    all_data.append({
                        'text': row['text'],
                        # Optional: keep metadata if needed
                        # 'id': row.get('id', ''),
                        # 'score': row.get('score', 0.0),
                        # 'token_count': row.get('token_count', 0)
                    })
                
                print(f"  Loaded {initial_len} examples")
                total_examples += initial_len
            else:
                print(f"  WARNING: No 'text' column found, skipping file")
            
            # Free memory
            del df
            gc.collect()
            
        except Exception as e:
            print(f"  ERROR loading file: {e}")
            continue
    
    if not all_data:
        print("ERROR: No valid data loaded!")
        sys.exit(1)
    
    print(f"\n" + "=" * 60)
    print(f"Total examples loaded: {total_examples:,}")
    print("=" * 60)
    
    # Create HuggingFace dataset
    print("\nCreating HuggingFace Dataset...")
    dataset = Dataset.from_list(all_data)
    
    # Save to disk
    save_path = "/workspace/fineweb_clean"
    print(f"Saving dataset to {save_path}...")
    dataset.save_to_disk(save_path)
    
    print("\n" + "=" * 60)
    print("SUCCESS! Dataset ready for training")
    print("=" * 60)
    print(f"Dataset saved to: {save_path}")
    print(f"Total examples: {len(dataset):,}")
    
    # Print sample
    if len(dataset) > 0:
        print("\nSample text (first 500 chars):")
        print("-" * 40)
        print(dataset[0]['text'][:500])
        print("-" * 40)
    
    # Create a simple config for this dataset
    config_path = "configs/fineweb_local.yaml"
    print(f"\nCreating config at {config_path}...")
    
    config_content = f"""# Auto-generated config for local FineWeb-Edu dataset
stage:
  name: "fineweb_edu_local"
  
data:
  dataset_name: "{save_path}"  # Local path to cleaned dataset
  dataset_split: "train"
  streaming: false
  max_sequence_length: 2048
  stride: 1024
  tokenizer: "gpt2"
  
training:
  num_epochs: 1
  max_steps: 150000
  per_device_train_batch_size: 384
  gradient_accumulation_steps: 1
  
  learning_rate: 8e-4
  min_learning_rate: 8e-5
  lr_scheduler_type: "cosine"
  warmup_steps: 500
  
  use_curriculum: true
  curriculum_stages:
    - {{step: 0,     seq_len: 128,  batch_size: 384, lr_scale: 10.0, grad_clip: 5.0}}
    - {{step: 3000,  seq_len: 256,  batch_size: 192, lr_scale: 6.0,  grad_clip: 2.0}}
    - {{step: 6000,  seq_len: 512,  batch_size: 96,  lr_scale: 3.0,  grad_clip: 1.0}}
    - {{step: 10000, seq_len: 768,  batch_size: 64,  lr_scale: 2.0,  grad_clip: 0.5}}
    - {{step: 15000, seq_len: 1024, batch_size: 48,  lr_scale: 1.5,  grad_clip: 0.4}}
    - {{step: 25000, seq_len: 1536, batch_size: 32,  lr_scale: 1.0,  grad_clip: 0.3}}
    - {{step: 40000, seq_len: 2048, batch_size: 24,  lr_scale: 0.7,  grad_clip: 0.3}}
  
  optimizer: "adamw"
  adam_beta1: 0.9
  adam_beta2: 0.95
  weight_decay: 0.1
  
  fp8_enabled: true
  bf16: true
  fp16: false
  
  save_strategy: "steps"
  save_steps: 10000
  save_total_limit: 3
  
  evaluation_strategy: "steps"
  eval_steps: 2000
  
  logging_steps: 100
  report_to: ["wandb"]
  seed: 42
  
validation:
  validation_split: 0.05
  per_device_eval_batch_size: 48
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Config created at: {config_path}")
    print("\n" + "=" * 60)
    print("TO START TRAINING:")
    print("=" * 60)
    print("Run this command:")
    print(f"  python train_fp8.py --config {config_path}")
    print("\nOr if FP8 doesn't work:")
    print(f"  python train_curriculum.py --config {config_path}")
    print("=" * 60)
    
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and clean FineWeb-Edu dataset")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of parquet files to load (for testing)")
    args = parser.parse_args()
    
    dataset = load_fineweb_dataset(max_files=args.max_files)
    print(f"\nDataset object: {dataset}")
    print("Done!")