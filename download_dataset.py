#!/usr/bin/env python3
"""
Download dataset with proper rate limit handling and retry logic.
This script downloads the dataset more efficiently than the default HF datasets loader.
"""

import os
import time
import sys
from pathlib import Path
from datasets import load_dataset, config
from huggingface_hub import snapshot_download
import argparse

def download_with_retry(dataset_name: str, max_retries: int = 10, base_delay: int = 10):
    """
    Download dataset with exponential backoff for rate limits.
    """
    print(f"Starting download of {dataset_name}")
    print("This will handle rate limits better than the default loader")
    
    # Method 1: Try using huggingface_hub's snapshot_download (better for many files)
    try:
        print("\nAttempting optimized download using snapshot_download...")
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        
        # Download all files at once with better rate limit handling
        local_dir = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            cache_dir=cache_dir,
            resume_download=True,  # Resume if interrupted
            max_workers=2,  # Limit concurrent downloads to avoid rate limits
        )
        print(f"Dataset downloaded to: {local_dir}")
        
        # Now load it normally (will use the cached version)
        print("\nLoading dataset from cache...")
        dataset = load_dataset(dataset_name, streaming=False)
        print(f"Successfully loaded dataset with {len(dataset['train'])} examples")
        return dataset
        
    except Exception as e:
        print(f"Snapshot download failed: {e}")
        print("Falling back to standard download with retry logic...")
    
    # Method 2: Fallback to standard download with better retry logic
    for attempt in range(max_retries):
        try:
            # Set longer timeout and retry settings
            config.DATASETS_DOWNLOAD_TIMEOUT = 60  # 60 second timeout
            
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            dataset = load_dataset(
                dataset_name,
                streaming=False,
                download_mode="force_redownload" if attempt > 0 else None,
                num_proc=1,  # Single process to avoid rate limits
            )
            
            print(f"Successfully downloaded dataset with {len(dataset['train'])} examples")
            return dataset
            
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                # Exponential backoff for rate limits
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                print(f"Error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {base_delay} seconds...")
                    time.sleep(base_delay)
                else:
                    raise

def download_dataset_files_individually(dataset_name: str):
    """
    Alternative: Download files one by one with delays to avoid rate limits.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    
    print(f"\nDownloading {dataset_name} files individually with rate limit handling...")
    
    # List all files in the dataset
    files = list_repo_files(repo_id=dataset_name, repo_type="dataset")
    jsonl_files = [f for f in files if f.endswith('.jsonl')]
    
    print(f"Found {len(jsonl_files)} JSONL files to download")
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / dataset_name.replace("/", "--")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file in enumerate(jsonl_files, 1):
        try:
            print(f"Downloading {i}/{len(jsonl_files)}: {file}")
            local_path = hf_hub_download(
                repo_id=dataset_name,
                filename=file,
                repo_type="dataset",
                cache_dir=cache_dir,
                resume_download=True
            )
            
            # Small delay between files to avoid rate limits
            if i < len(jsonl_files):
                time.sleep(0.5)  # 500ms delay between files
                
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limited on file {file}. Waiting 30 seconds...")
                time.sleep(30)
                # Retry this file
                local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=file,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                    resume_download=True
                )
    
    print(f"\nAll files downloaded to {cache_dir}")
    print("Now loading dataset from cached files...")
    
    # Load from cache
    dataset = load_dataset(dataset_name, streaming=False)
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Download HuggingFace dataset with rate limit handling')
    parser.add_argument('--dataset', type=str, default='Yxanul/wikipedia-2k-high-quality',
                        help='Dataset name to download')
    parser.add_argument('--method', type=str, choices=['auto', 'snapshot', 'individual'], 
                        default='auto', help='Download method to use')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HuggingFace Dataset Downloader")
    print("=" * 60)
    
    try:
        if args.method == 'individual':
            dataset = download_dataset_files_individually(args.dataset)
        elif args.method == 'snapshot':
            dataset = download_with_retry(args.dataset)
        else:  # auto
            dataset = download_with_retry(args.dataset)
        
        print("\n" + "=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Dataset: {args.dataset}")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Columns: {dataset['train'].column_names}")
        print("\nDataset is now cached and ready for training!")
        
    except Exception as e:
        print(f"\nFailed to download dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Try again in a few minutes (rate limits reset)")
        print("3. Use --method=individual for more granular control")
        print("4. Consider using HuggingFace CLI: huggingface-cli download")
        sys.exit(1)

if __name__ == "__main__":
    main()