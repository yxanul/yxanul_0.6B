"""
Data Pipeline for Yxanul 0.6B Training
Implements efficient streaming data loading from HuggingFace datasets.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
from typing import Dict, Optional, Iterator
import random

# Maximum buffer size to prevent memory leaks
MAX_BUFFER_SIZE = 100000  # Prevent OOM with very long documents


class StreamingDataset(IterableDataset):
    """Streaming dataset for efficient data loading without downloading entire dataset."""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_length: int = 2048,
        stride: int = 1024,
        stage_config: Dict = None,
        split: str = "train"
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.stage_config = stage_config or {}
        self.split = split
        
        # For sequence length curriculum - use training config
        training_config = stage_config.get("training", {})
        if training_config.get("use_curriculum", False) and training_config.get("curriculum_stages"):
            # Get the first stage's sequence length
            self.current_seq_length = training_config["curriculum_stages"][0]["seq_len"]
        else:
            self.current_seq_length = max_length
        self.target_seq_length = max_length
        self.curriculum_stages = training_config.get("curriculum_stages", [])
        
        # Load the dataset (will download and cache on first run)
        print(f"Loading dataset: {self.dataset_name} (split: {self.split})")
        
        # Try loading with better error handling for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.dataset = load_dataset(
                    self.dataset_name,
                    split=self.split,
                    streaming=False,  # Download the full dataset
                    num_proc=1  # Single process to avoid rate limits
                )
                print(f"Dataset loaded: {len(self.dataset)} examples")
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    import time
                    time.sleep(wait_time)
                else:
                    print(f"Error loading dataset: {e}")
                    print("\nPlease run 'python download_dataset.py' first to download the dataset")
                    print("This will handle rate limits better than the default loader")
                    raise
        
    def __len__(self):
        """Return estimated number of chunks."""
        # Rough estimate: ~2 chunks per document
        return len(self.dataset) * 2
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset."""
        # Shuffle if in training mode
        if "train" in self.split:
            dataset = self.dataset.shuffle(seed=42)
        else:
            dataset = self.dataset
        
        # Process each example
        buffer = []
        for example in dataset:
            # Get text from the example (handle different formats)
            if "text" in example:
                text = example["text"]
            elif "content" in example:
                text = example["content"]
            elif "article" in example:
                text = example["article"]
            else:
                # Skip if no text field found
                continue
                
            # Tokenize without truncation (we handle chunking ourselves)
            tokens = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=False
            )["input_ids"]
            
            # Add to buffer
            buffer.extend(tokens)
            
            # Prevent memory leak - truncate buffer if too large
            if len(buffer) > MAX_BUFFER_SIZE:
                # Keep only the last MAX_BUFFER_SIZE tokens
                print(f"Warning: Buffer exceeded {MAX_BUFFER_SIZE} tokens, truncating to prevent OOM")
                buffer = buffer[-MAX_BUFFER_SIZE:]
            
            # Create chunks when buffer is large enough
            while len(buffer) >= self.current_seq_length:
                chunk = buffer[:self.current_seq_length]
                buffer = buffer[self.stride:]  # Sliding window with stride
                
                # Create input and labels
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                
                # Pad if necessary
                if len(input_ids) < self.current_seq_length - 1:
                    pad_length = self.current_seq_length - 1 - len(input_ids)
                    input_ids = torch.cat([
                        input_ids,
                        torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                    ])
                    labels = torch.cat([
                        labels,
                        torch.full((pad_length,), -100, dtype=torch.long)  # -100 is ignored in loss
                    ])
                
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": (input_ids != self.tokenizer.pad_token_id).long()
                }
    
    def update_sequence_length(self, step: int, total_steps: int):
        """Update sequence length for curriculum learning."""
        if self.curriculum_stages:
            # Find the appropriate stage based on current step
            for stage in self.curriculum_stages:
                if step >= stage.get("step", 0):
                    self.current_seq_length = stage["seq_len"]
                else:
                    break
            return self.current_seq_length
        return self.max_length


class DataCollator:
    """Custom data collator for batching."""
    
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        """Collate batch of features."""
        # Stack all tensors
        batch = {}
        for key in features[0].keys():
            if key == "labels":
                # Labels use -100 for padding (ignored in loss)
                batch[key] = torch.stack([f[key] for f in features])
            else:
                batch[key] = torch.stack([f[key] for f in features])
        
        return batch


def create_dataloader(
    dataset_name: str,
    tokenizer,
    batch_size: int,
    max_length: int = 2048,
    stage_config: Dict = None,
    num_workers: int = 2,
    split: str = "train"
) -> DataLoader:
    """Create a DataLoader for streaming dataset."""
    
    dataset = StreamingDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=int(max_length * 0.5),  # 50% overlap
        stage_config=stage_config,
        split=split
    )
    
    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return dataloader, dataset


def estimate_dataset_size(dataset_name: str, split: str = "train") -> int:
    """Estimate the number of examples in a dataset."""
    # Known dataset sizes (approximate)
    known_sizes = {
        "Yxanul/wikipedia-2k-high-quality": 239_000,
        "open-r1/Mixture-of-Thoughts": 350_000,
        "gsm8k": 7_500,
        "bigcode/starcoderdata": 1_000_000,  # Sample
        "euclaise/TinyCoT": 147_000,
    }
    
    if dataset_name in known_sizes:
        return known_sizes[dataset_name]
    
    # Try to get actual size
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=False)
        return len(dataset)
    except:
        # Default estimate
        return 100_000


def create_tokenizer(model_name: str = "gpt2") -> GPT2Tokenizer:
    """Create and configure tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set max length
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


# Multi-dataset mixing for later stages
class MixedDataset(IterableDataset):
    """Mix multiple datasets with specified weights."""
    
    def __init__(
        self,
        datasets_config: list,
        tokenizer,
        max_length: int = 2048,
        mixing_weights: list = None
    ):
        self.datasets = []
        self.weights = mixing_weights or [1.0] * len(datasets_config)
        
        for config in datasets_config:
            dataset = StreamingDataset(
                dataset_name=config["name"],
                tokenizer=tokenizer,
                max_length=max_length,
                stage_config=config.get("stage_config", {})
            )
            self.datasets.append(dataset)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def __iter__(self):
        """Iterate through mixed datasets."""
        iterators = [iter(d) for d in self.datasets]
        
        while True:
            # Sample dataset based on weights
            dataset_idx = np.random.choice(len(self.datasets), p=self.weights)
            
            try:
                yield next(iterators[dataset_idx])
            except StopIteration:
                # Restart this dataset
                iterators[dataset_idx] = iter(self.datasets[dataset_idx])
                yield next(iterators[dataset_idx])