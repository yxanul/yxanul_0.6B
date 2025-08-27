"""
Simplified Data Pipeline for Yxanul Training
NO STREAMING - Only local data loading
Optimized for experimental-pretrain-1b dataset
"""

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, Optional, Iterator
from pathlib import Path
import os

class LocalPretrainDataset(Dataset):
    """Simple dataset for local parquet files - NO STREAMING.
    
    Expects experimental-pretrain-1b format:
    - text (string): The text content
    - source (string): 'fineweb', 'math', or 'code'
    - num_tokens (int64): Pre-computed token count
    """
    
    def __init__(
        self,
        dataset_path: str = "./experimental-pretrain-1b/dataset_1b.parquet",
        tokenizer=None,
        max_length: int = 2048,
        split_range: tuple = None  # e.g., (0, 0.9) for first 90%
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading LOCAL dataset from: {dataset_path}")
        
        # Find the parquet file
        if os.path.exists(dataset_path):
            parquet_path = dataset_path
        else:
            # Try common locations
            paths_to_try = [
                Path("experimental-pretrain-1b/dataset_1b.parquet"),
                Path("/workspace/yxanul_0.6B/experimental-pretrain-1b/dataset_1b.parquet"),
                Path("./dataset_1b.parquet"),
            ]
            parquet_path = None
            for p in paths_to_try:
                if p.exists():
                    parquet_path = str(p)
                    print(f"Found dataset at: {parquet_path}")
                    break
            
            if not parquet_path:
                raise FileNotFoundError(f"Cannot find dataset_1b.parquet. Tried: {paths_to_try}")
        
        # Load the entire dataset into memory
        print("Loading parquet file...")
        self.dataset = load_dataset("parquet", data_files=parquet_path, split="train")
        
        # Apply split if specified
        if split_range:
            start_idx = int(len(self.dataset) * split_range[0])
            end_idx = int(len(self.dataset) * split_range[1])
            self.dataset = self.dataset.select(range(start_idx, end_idx))
            print(f"Using samples {start_idx} to {end_idx} ({end_idx - start_idx} total)")
        
        print(f"Dataset loaded: {len(self.dataset)} examples")
        
        # Show statistics
        if 'num_tokens' in self.dataset.features:
            total_tokens = sum(self.dataset['num_tokens'])
            print(f"Total pre-computed tokens: {total_tokens:,}")
            print(f"Average tokens per doc: {total_tokens / len(self.dataset):.1f}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a single example and tokenize it."""
        example = self.dataset[idx]
        text = example['text']
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels (mask padding)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def create_dataloader(
    dataset_name: str = "Yxanul/experimental-pretrain-1b",
    tokenizer=None,
    batch_size: int = 1,
    max_length: int = 2048,
    num_workers: int = 2,
    split: str = "train[:90%]",
    **kwargs  # Ignore extra arguments for compatibility
) -> tuple:
    """Create a DataLoader for LOCAL dataset only - NO STREAMING.
    
    Args:
        dataset_name: Should be "Yxanul/experimental-pretrain-1b" or path to local file
        tokenizer: Tokenizer to use
        batch_size: Batch size (will be adjusted to multiple of 8 for FP8)
        max_length: Maximum sequence length
        num_workers: DataLoader workers
        split: Dataset split (e.g., "train[:90%]" for first 90%)
    """
    
    # Parse split to get range
    split_range = None
    if "[" in split and "]" in split:
        # Parse splits like "train[:90%]" or "train[90%:]"
        import re
        if ":" in split:
            parts = split[split.index("[")+1:split.index("]")].split(":")
            start = 0 if not parts[0] else float(parts[0].strip("%")) / 100
            end = 1.0 if not parts[1] else float(parts[1].strip("%")) / 100
            split_range = (start, end)
    
    # Always use local dataset
    if "yxanul" in dataset_name.lower() or "experimental" in dataset_name.lower():
        dataset_path = "./experimental-pretrain-1b/dataset_1b.parquet"
    else:
        dataset_path = dataset_name  # Assume it's a path
    
    print(f"Creating LOCAL dataset (NO STREAMING)")
    dataset = LocalPretrainDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split_range=split_range
    )
    
    # Adjust batch size for FP8 (must be multiple of 8)
    if batch_size % 8 != 0 and batch_size > 1:
        old_batch = batch_size
        batch_size = max(8, (batch_size // 8) * 8)
        if batch_size == 0:
            batch_size = 1  # Fallback to 1
        print(f"Adjusted batch size from {old_batch} to {batch_size} for FP8")
    
    # Simple collate function
    def collate_fn(batch):
        input_ids = torch.stack([x['input_ids'] for x in batch])
        labels = torch.stack([x['labels'] for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=("train" in split),  # Shuffle for training
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader, dataset


def create_tokenizer(model_name: str = None, use_superbpe: bool = True):
    """Create tokenizer (SuperBPE by default)."""
    if use_superbpe:
        print("Loading SuperBPE tokenizer for efficiency...")
        
        # Try local cache first
        cache_paths = ["./tokenizer_cache/superbpe-t80k-fast", "./tokenizer_cache/superbpe-t180k-fast"]
        for cache_path in cache_paths:
            if os.path.exists(cache_path):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(cache_path)
                    print(f"Loaded from cache: {cache_path}")
                    break
                except:
                    pass
        else:
            # Load from HuggingFace
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "UW/OLMo2-8B-SuperBPE-t80k",
                    trust_remote_code=True,
                    use_fast=True
                )
                print("SuperBPE tokenizer loaded from HuggingFace")
            except:
                print("Falling back to GPT-2 tokenizer")
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name or "gpt2")
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.model_max_length = 4096
    return tokenizer


def calculate_training_steps(
    dataset_size: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1
) -> int:
    """Calculate total training steps."""
    effective_batch_size = batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = dataset_size // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    return total_steps


def estimate_dataset_size(dataset_name: str, split: str = "train") -> int:
    """Estimate the number of examples in a dataset."""
    # For experimental-pretrain-1b, we know the size
    if "experimental-pretrain" in dataset_name.lower():
        return 605_376  # Actual size of the dataset
    
    # Try to load and check
    try:
        if "yxanul" in dataset_name.lower():
            dataset_path = "./experimental-pretrain-1b/dataset_1b.parquet"
            dataset = load_dataset("parquet", data_files=dataset_path, split="train")
            return len(dataset)
    except:
        pass
    
    # Default estimate
    return 100_000


# For backward compatibility with imports
def create_curriculum_dataloader(
    curriculum_config: dict,
    tokenizer=None,
    batch_size: int = 1,
    max_length: int = None,  # Will be set from curriculum stage
    num_workers: int = 2,
    current_stage: int = 0,
    **kwargs
) -> tuple:
    """Create a DataLoader with curriculum-specific settings.
    
    Args:
        curriculum_config: Full curriculum configuration dict
        tokenizer: Tokenizer to use
        batch_size: Batch size (from curriculum stage)
        max_length: Override sequence length (if None, uses stage seq_len)
        num_workers: DataLoader workers
        current_stage: Current curriculum stage index
    """
    
    # Get current stage configuration
    stages = curriculum_config['training']['curriculum_stages']
    if current_stage >= len(stages):
        current_stage = len(stages) - 1
    
    stage_config = stages[current_stage]
    
    # Use stage-specific sequence length unless overridden
    if max_length is None:
        max_length = stage_config.get('seq_len', 2048)
    
    # Use stage-specific batch size if not provided
    if 'batch_size' in stage_config:
        batch_size = stage_config['batch_size']
    
    print(f"\nCurriculum Stage {current_stage + 1}/{len(stages)}: {stage_config.get('name', 'Unknown')}")
    print(f"  Sequence length: {max_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target tokens: {stage_config.get('tokens', 'Not specified'):,}")
    
    # Get dataset configuration
    dataset_name = curriculum_config['training'].get('dataset', 'Yxanul/experimental-pretrain-1b')
    dataset_path = curriculum_config['training'].get('dataset_path', 'experimental-pretrain-1b')
    dataset_file = curriculum_config['training'].get('dataset_file', 'dataset_1b.parquet')
    
    # Construct full path
    full_path = f"./{dataset_path}/{dataset_file}"
    
    # Create regular dataloader with curriculum-specific settings
    return create_dataloader(
        dataset_name=full_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        split="train",  # Always use full dataset for curriculum
        **kwargs
    )


class MixedDataset(Dataset):
    """Compatibility stub for mixed datasets."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MixedDataset not supported in simple version")


class CurriculumStreamingDataset(Dataset):
    """Compatibility stub for curriculum datasets."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Curriculum streaming not supported - use local data only")


class DataCollator:
    """Simple data collator for batching."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in features[0].keys():
            if key == "labels":
                batch[key] = torch.stack([f[key] for f in features])
            else:
                batch[key] = torch.stack([f[key] for f in features])
        return batch


def fp8_collate_fn(batch):
    """Simple collate function for FP8."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


# Compatibility exports
PretrainDataset = LocalPretrainDataset
StreamingDataset = LocalPretrainDataset  # No streaming!
IterableDataset = Dataset  # No streaming!