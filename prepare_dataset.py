#!/usr/bin/env python3
"""
Prepare the Wikipedia dataset for training.

IMPORTANT: You must first clone the dataset using git:
  git clone https://huggingface.co/datasets/Yxanul/wikipedia-2k-high-quality ./data/wikipedia

This script then processes the JSONL files into a format optimized for training.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict
import requests
from tqdm import tqdm
from datasets import Dataset, DatasetDict

def download_dataset_files(dataset_name: str = "Yxanul/wikipedia-2k-high-quality", 
                          output_dir: str = "./data/wikipedia"):
    """
    Download dataset files directly using git or wget to avoid rate limits.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Dataset Downloader")
    print("=" * 60)
    
    # Option 1: Use git to clone the entire dataset (RECOMMENDED)
    print("\nOption 1: Clone with git (Recommended)")
    print("Run this command in your terminal:")
    print(f"\ngit clone https://huggingface.co/datasets/{dataset_name} {output_dir}")
    print("\nThis will download all files at once without rate limits!")
    
    # Option 2: Download individual files with wget
    print("\n" + "-" * 60)
    print("Option 2: Download with wget/curl")
    print("Create a script with these commands:\n")
    
    base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
    
    # Generate wget commands for batch files (skip batch_0000 which has different columns)
    for i in range(1, 240):  # Assuming 239 files based on your logs
        file_name = f"wikipedia_2k_batch_{i:04d}.jsonl"
        print(f"wget {base_url}/{file_name} -P {output_dir}/")
    
    print("\n" + "=" * 60)
    return output_path

def load_local_jsonl_files(data_dir: str, skip_first: bool = True) -> Dataset:
    """
    Load JSONL files from local directory, skipping the metadata file.
    This avoids all HuggingFace API calls and rate limits.
    """
    data_path = Path(data_dir)
    
    # Find all JSONL files - try different patterns
    jsonl_files = list(data_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {data_path}")
        print(f"Contents of directory: {list(data_path.iterdir())[:5]}")
        return None
    
    # Filter out batch_0000 if it exists
    if skip_first:
        jsonl_files = [f for f in jsonl_files if "batch_0000" not in f.name and "batch_000.jsonl" not in f.name]
    
    # Sort the files
    jsonl_files = sorted(jsonl_files)
    
    print(f"Found {len(jsonl_files)} JSONL files to load")
    if len(jsonl_files) > 0:
        print(f"First file: {jsonl_files[0].name}")
        print(f"Last file: {jsonl_files[-1].name}")
    
    # Load all data into memory
    all_data = []
    files_processed = 0
    
    for file_path in tqdm(jsonl_files, desc="Loading files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_lines = 0
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Only keep the columns we need
                        filtered_data = {
                            'text': data.get('text', ''),
                            'title': data.get('title', data.get('extracted_title', '')),
                            'token_count': data.get('token_count', 0)
                        }
                        if filtered_data['text']:  # Only add if there's actual text
                            all_data.append(filtered_data)
                            file_lines += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        continue
                
                if file_lines > 0:
                    files_processed += 1
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"Processed {files_processed} files")
    print(f"Loaded {len(all_data)} examples total")
    
    if len(all_data) == 0:
        print("WARNING: No data loaded! Check file format.")
        return None
    
    # Create HuggingFace Dataset from the data
    dataset = Dataset.from_list(all_data)
    return dataset

def create_train_val_split(dataset: Dataset, val_split: float = 0.05) -> DatasetDict:
    """
    Create train/validation split from the dataset.
    """
    # Shuffle and split
    dataset = dataset.shuffle(seed=42)
    
    split_idx = int(len(dataset) * (1 - val_split))
    
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(dataset)))
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

def prepare_dataset_for_training(data_dir: str = "./data/wikipedia"):
    """
    Main function to prepare the dataset for training.
    """
    print("Preparing dataset for training...")
    
    # Check multiple possible locations
    possible_paths = [
        Path(data_dir),
        Path(data_dir) / "wikipedia-2k-high-quality",  # Handle nested git clone
        Path("./wikipedia-2k-high-quality"),  # Direct clone
    ]
    
    actual_path = None
    for path in possible_paths:
        if path.exists() and list(path.glob("*.jsonl")):
            actual_path = path
            print(f"Found dataset at: {actual_path}")
            break
    
    if not actual_path:
        print(f"\nDataset not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease download the dataset first using:")
        print(f"  git clone https://huggingface.co/datasets/Yxanul/wikipedia-2k-high-quality")
        return None
    
    data_dir = str(actual_path)
    
    # Load the dataset from local files
    dataset = load_local_jsonl_files(data_dir)
    
    # Check if dataset was loaded successfully
    if dataset is None or len(dataset) == 0:
        print("\nERROR: No data was loaded from the JSONL files!")
        print("\nPossible issues:")
        print("1. The JSONL files might be empty")
        print("2. The files might have a different format than expected")
        print("3. The 'text' field might be missing from the files")
        print("\nTry checking a sample file:")
        print(f"  head -1 {data_dir}/*.jsonl | python -m json.tool")
        return None
    
    # Create train/val split
    dataset_dict = create_train_val_split(dataset)
    
    print(f"\nDataset prepared:")
    print(f"Train examples: {len(dataset_dict['train'])}")
    print(f"Validation examples: {len(dataset_dict['validation'])}")
    
    # Only save if we have data
    if len(dataset_dict['train']) > 0:
        # Save to disk for faster loading next time
        cache_dir = Path("./data/processed_dataset")
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(cache_dir)
        print(f"\nDataset saved to {cache_dir} for faster loading")
    else:
        print("\nWARNING: Dataset is empty, not saving to disk")
    
    return dataset_dict

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        # Show download instructions
        download_dataset_files()
    else:
        # Prepare dataset for training
        dataset = prepare_dataset_for_training()
        
        if dataset:
            print("\n" + "=" * 60)
            print("Dataset ready for training!")
            print("=" * 60)
            print("\nTo use this dataset in training, update data_pipeline.py to load from local files")