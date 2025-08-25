"""
Data Pipeline for Yxanul 0.6B Training
Implements efficient streaming data loading from HuggingFace datasets.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, Optional, Iterator
import random
from pathlib import Path
import os  # Required for environment variables

# Maximum buffer size to prevent memory leaks
MAX_BUFFER_SIZE = 100000  # Prevent OOM with very long documents


class StreamingDataset(IterableDataset):
    """Streaming dataset for efficient data loading without downloading entire dataset.
    
    This dataset properly handles document boundaries by inserting EOS tokens between
    documents, which is crucial for language model training to learn when to stop
    generating and to avoid treating concatenated documents as continuous text.
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_length: int = 2048,
        stride: int = None,  # Will be computed based on current sequence length
        stage_config: Dict = None,
        split: str = "train",
        add_eos_between_docs: bool = True
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage_config = stage_config if stage_config is not None else {}
        self.split = split
        self.add_eos_between_docs = add_eos_between_docs  # Add EOS tokens at document boundaries
        
        # For sequence length curriculum - use training config
        training_config = self.stage_config.get("training", {})
        self.curriculum_stages = training_config.get("curriculum_stages", [])
        
        # Use the provided max_length as current sequence length
        # This ensures when dataloader is recreated with new max_length, it uses that value
        self.current_seq_length = max_length
        self.target_seq_length = max_length
        
        # Only use first stage seq_len if max_length wasn't explicitly set to something else
        if training_config.get("use_curriculum", False) and self.curriculum_stages:
            # If max_length is still the default (2048), use first stage
            if max_length == 2048 and self.curriculum_stages:
                self.current_seq_length = self.curriculum_stages[0]["seq_len"]
        
        # Compute stride based on current sequence length (50% overlap for efficiency)
        # For very short sequences, use full sequence as stride (no overlap)
        if stride is None:
            if self.current_seq_length <= 64:
                self.stride = self.current_seq_length  # No overlap for very short sequences
            else:
                self.stride = max(1, int(self.current_seq_length * 0.5))  # 50% overlap
        else:
            self.stride = stride
        
        print(f"Dataset initialized with seq_len={self.current_seq_length}, stride={self.stride}")
        
        # Load the dataset - try local first, then HuggingFace
        print(f"Loading dataset: {self.dataset_name} (split: {self.split})")
        
        # Check for local preprocessed dataset first
        local_dataset_path = Path("./data/processed_dataset")
        local_raw_path = Path("./data/wikipedia")
        
        # Check multiple possible locations for cloned dataset
        possible_clone_paths = [
            Path("./fineweb-edu-highest-quality-2025"),
            Path("/workspace/fineweb-edu-highest-quality-2025"),
            Path("./data/fineweb-edu-highest-quality-2025"),
            Path("D:/ai_testing/yxanul_0.6B/fineweb-edu-highest-quality-2025"),
            Path("D:/ai_testing/fineweb-edu-highest-quality-2025"),
        ]
        
        local_cloned_path = None
        for path in possible_clone_paths:
            if path.exists():
                local_cloned_path = path
                break
        
        if local_dataset_path.exists():
            # Load preprocessed dataset from disk
            print(f"Loading preprocessed dataset from {local_dataset_path}")
            from datasets import load_from_disk
            dataset_dict = load_from_disk(local_dataset_path)
            
            # Handle split selection
            if "train[" in self.split:
                # Parse percentage split like "train[95%:]"
                import re
                match = re.match(r"train\[(\d+)%:\]", self.split)
                if match:
                    start_pct = int(match.group(1))
                    split_idx = int(len(dataset_dict['train']) * (start_pct / 100))
                    self.dataset = dataset_dict['train'].select(range(split_idx, len(dataset_dict['train'])))
                else:
                    self.dataset = dataset_dict['train']
            elif self.split in dataset_dict:
                self.dataset = dataset_dict[self.split]
            else:
                self.dataset = dataset_dict['train']
            print(f"Dataset loaded from local: {len(self.dataset)} examples")
            
        elif local_raw_path.exists():
            # Load from raw JSONL files
            print(f"Loading raw JSONL files from {local_raw_path}")
            import json
            from datasets import Dataset
            
            # Load all JSONL files except batch_0000 (metadata)
            all_data = []
            jsonl_files = sorted(local_raw_path.glob("wikipedia_2k_batch_*.jsonl"))
            jsonl_files = [f for f in jsonl_files if "batch_0000" not in f.name]
            
            for file_path in jsonl_files:  # Load all files
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            all_data.append({
                                'text': data.get('text', ''),
                                'title': data.get('title', ''),
                                'token_count': data.get('token_count', 0)
                            })
                        except:
                            continue
            
            self.dataset = Dataset.from_list(all_data)
            print(f"Dataset loaded from JSONL: {len(self.dataset)} examples")
        elif local_cloned_path and local_cloned_path.exists() and "fineweb" in self.dataset_name.lower():
            # Load from cloned FineWeb-Edu repository
            print(f"Loading FineWeb-Edu from cloned repository at {local_cloned_path}")
            
            # Store path to parquet files for lazy loading
            parquet_files = sorted(list(local_cloned_path.glob("data/*.parquet")))
            if not parquet_files:
                parquet_files = sorted(list(local_cloned_path.glob("*.parquet")))
            
            if parquet_files:
                print(f"Found {len(parquet_files)} parquet files")
                
                # Split files for train/validation
                if "validation" in self.split or "val" in self.split:
                    val_count = max(1, len(parquet_files) // 20)
                    self.parquet_files = parquet_files[-val_count:]
                    print(f"Using {len(self.parquet_files)} files for validation")
                else:
                    train_count = len(parquet_files) - max(1, len(parquet_files) // 20)
                    self.parquet_files = parquet_files[:train_count]
                    print(f"Using {len(self.parquet_files)} files for training")
                
                # Store for lazy loading during iteration
                self.dataset = None  # We'll load lazily
                self.is_lazy_parquet = True
            else:
                raise ValueError(f"No parquet files found in {local_cloned_path}")
            
        else:
            # Fallback to HuggingFace (with rate limit handling)
            print("Local dataset not found. Attempting to download from HuggingFace...")
            print("This may fail due to rate limits. Consider running:")
            print(f"  git clone https://huggingface.co/datasets/{self.dataset_name} /workspace/{self.dataset_name.split('/')[-1]}")
            print("  Then restart training")
            
            try:
                # Check if it's a local path
                if self.dataset_name.startswith('/') or self.dataset_name.startswith('./'):
                    from datasets import load_from_disk
                    self.dataset = load_from_disk(self.dataset_name)
                    print(f"Loaded local dataset from {self.dataset_name}")
                else:
                    # Load from HuggingFace
                    self.dataset = load_dataset(
                        self.dataset_name,
                        split=self.split,
                        streaming=False,
                        num_proc=1
                    )
                print(f"Dataset loaded: {len(self.dataset)} examples")
            except Exception as e:
                print(f"\nError: {e}")
                print("\n" + "=" * 60)
                print("SOLUTION: Download the dataset locally first:")
                print("=" * 60)
                print("Run this command:")
                print(f"  git clone https://huggingface.co/datasets/{self.dataset_name} ./data/wikipedia")
                print("\nThen run:")
                print("  python download_and_prepare_dataset.py")
                print("=" * 60)
                raise
        
    def __len__(self):
        """Return estimated number of chunks."""
        if hasattr(self, 'is_lazy_parquet') and self.is_lazy_parquet:
            # Estimate based on number of files and avg samples per file
            return len(self.parquet_files) * 10000  # Rough estimate
        elif self.dataset is not None:
            # Rough estimate: ~2 chunks per document
            return len(self.dataset) * 2
        else:
            return 0
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset."""
        buffer = []
        
        # Handle lazy parquet loading
        if hasattr(self, 'is_lazy_parquet') and self.is_lazy_parquet:
            import pandas as pd
            import random
            
            # Shuffle files if training
            files = list(self.parquet_files)
            if "train" in self.split:
                random.shuffle(files)
            
            # Process files one by one
            for file_idx, parquet_file in enumerate(files):
                if file_idx % 10 == 0:
                    print(f"Processing file {file_idx + 1}/{len(files)}: {parquet_file.name}")
                
                try:
                    # Load one file at a time
                    df = pd.read_parquet(parquet_file)
                    
                    # Get text column
                    text_column = None
                    for col in ['text', 'content', 'document']:
                        if col in df.columns:
                            text_column = col
                            break
                    
                    if not text_column:
                        continue
                    
                    # Process texts from this file
                    texts = df[text_column].dropna()
                    texts = texts[texts.str.len() > 100]  # Filter short texts
                    
                    # Shuffle texts within file if training
                    text_list = texts.tolist()
                    if "train" in self.split:
                        random.shuffle(text_list)
                    
                    # Process each text
                    for text in text_list:
                        # Tokenize
                        tokens = self.tokenizer(
                            text,
                            truncation=False,
                            add_special_tokens=True,
                            return_attention_mask=False
                        )["input_ids"]
                        
                        # Add to buffer with EOS token at document boundary
                        buffer.extend(tokens)
                        # Add EOS token to mark document boundary (important for proper language modeling)
                        if self.add_eos_between_docs and self.tokenizer.eos_token_id is not None:
                            buffer.append(self.tokenizer.eos_token_id)
                        
                        # Prevent memory leak
                        if len(buffer) > MAX_BUFFER_SIZE:
                            buffer = buffer[-MAX_BUFFER_SIZE:]
                        
                        # Create chunks when buffer is large enough
                        while len(buffer) >= self.current_seq_length:
                            chunk = buffer[:self.current_seq_length]
                            # Use proper stride based on current sequence length
                            actual_stride = min(self.stride, self.current_seq_length)
                            buffer = buffer[actual_stride:]  # Sliding window
                            
                            # Create full-length input_ids and labels
                            # Labels are same as input_ids but with first position masked
                            input_ids = torch.tensor(chunk, dtype=torch.long)
                            labels = torch.tensor(chunk, dtype=torch.long)
                            labels[0] = -100  # Mask first position (no previous token to predict from)
                            
                            # No padding needed since we already have full sequence length
                            
                            yield {
                                "input_ids": input_ids,
                                "labels": labels,
                                "attention_mask": torch.ones(self.current_seq_length, dtype=torch.long)
                            }
                    
                except Exception as e:
                    print(f"Warning: Error processing {parquet_file.name}: {e}")
                    continue
        
        # Handle regular dataset
        elif self.dataset is not None:
            # Shuffle if in training mode
            if "train" in self.split:
                dataset = self.dataset.shuffle(seed=42)
            else:
                dataset = self.dataset
            
            # Process each example
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
                
                # Add to buffer with EOS token at document boundary
                buffer.extend(tokens)
                # Add EOS token to mark document boundary
                if self.tokenizer.eos_token_id is not None:
                    buffer.append(self.tokenizer.eos_token_id)
                
                # Prevent memory leak - truncate buffer if too large
                if len(buffer) > MAX_BUFFER_SIZE:
                    # Keep only the last MAX_BUFFER_SIZE tokens
                    print(f"Warning: Buffer exceeded {MAX_BUFFER_SIZE} tokens, truncating to prevent OOM")
                    buffer = buffer[-MAX_BUFFER_SIZE:]
                
                # Create chunks when buffer is large enough
                while len(buffer) >= self.current_seq_length:
                    chunk = buffer[:self.current_seq_length]
                    # Use proper stride based on current sequence length
                    actual_stride = min(self.stride, self.current_seq_length)
                    buffer = buffer[actual_stride:]  # Sliding window with stride
                    
                    # Create full-length input_ids and labels
                    # Labels are same as input_ids but with first position masked
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    labels = torch.tensor(chunk, dtype=torch.long)
                    labels[0] = -100  # Mask first position (no previous token to predict from)
                    
                    # No padding needed since we already have full sequence length
                    
                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": torch.ones(self.current_seq_length, dtype=torch.long)
                    }
    
    def update_sequence_length(self, step: int, total_steps: int):
        """Update sequence length for curriculum learning."""
        if self.curriculum_stages:
            # Find the appropriate stage based on current step
            old_seq_length = self.current_seq_length
            for stage in self.curriculum_stages:
                if step >= stage.get("step", 0):
                    self.current_seq_length = stage["seq_len"]
                else:
                    break
            
            # Update stride when sequence length changes
            if old_seq_length != self.current_seq_length:
                if self.current_seq_length <= 64:
                    self.stride = self.current_seq_length  # No overlap for very short sequences
                else:
                    self.stride = max(1, int(self.current_seq_length * 0.5))  # 50% overlap
                print(f"Updated seq_len={self.current_seq_length}, stride={self.stride} at step {step}")
            
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


def fp8_collate_fn(batch):
    """Simple collate function - no padding needed if batch sizes are multiples of 8."""
    # Stack the batch normally
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }

def create_dataloader(
    dataset_name: str,
    tokenizer,
    batch_size: int,
    max_length: int = 2048,
    stage_config: Dict = None,
    num_workers: int = 2,
    split: str = "train",
    add_eos_between_docs: bool = True
) -> DataLoader:
    """Create a DataLoader for streaming dataset."""
    
    # Don't pass stride - let StreamingDataset compute it based on sequence length
    dataset = StreamingDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=None,  # Will be computed based on current sequence length
        stage_config=stage_config,
        split=split,
        add_eos_between_docs=add_eos_between_docs
    )
    
    # Use FP8 collate function for proper dimension padding
    # FP8 requires batch dimension divisible by 8
    use_fp8_collate = os.environ.get("USE_FP8", "true").lower() == "true"
    
    if use_fp8_collate:
        collate_fn = fp8_collate_fn
    else:
        collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
        collate_fn = collator
    
    # Ensure batch size is multiple of 8 for FP8
    if batch_size % 8 != 0:
        print(f"Warning: Batch size {batch_size} not divisible by 8, adjusting to {(batch_size // 8) * 8}")
        batch_size = (batch_size // 8) * 8
        if batch_size == 0:
            batch_size = 8
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True  # Drop incomplete batches to maintain FP8 requirements
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


def create_tokenizer(model_name: str = "UW/OLMo2-8B-SuperBPE-t80k", use_superbpe: bool = True, max_length: int = None):
    """Create and configure tokenizer.
    
    Args:
        model_name: Tokenizer model name (defaults to SuperBPE t=80k for maximum efficiency)
        use_superbpe: Whether to use SuperBPE tokenizer (37.5% fewer tokens!)
        max_length: Maximum sequence length (if None, uses model's max_position_embeddings)
    """
    if use_superbpe:
        # Use SuperBPE-t80k tokenizer for 37.5% token reduction (maximum efficiency)
        print("Loading SuperBPE t=80k tokenizer (37.5% fewer tokens than GPT-2!)...")
        print("Research mode: Maximum compression for faster experimentation")
        
        # Try loading from local cache first
        cache_paths = ["./tokenizer_cache/superbpe-t80k", "./tokenizer_cache/superbpe-t180k"]
        tokenizer = None
        
        for cache_path in cache_paths:
            if os.path.exists(cache_path):
                try:
                    print(f"Loading tokenizer from cache: {cache_path}")
                    tokenizer = AutoTokenizer.from_pretrained(cache_path)
                    print(f"✓ Loaded from cache successfully!")
                    break
                except:
                    pass
        
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "UW/OLMo2-8B-SuperBPE-t80k",  # t=80k for maximum compression
                    token=os.environ.get("HF_TOKEN"),  # Use environment variable for security
                    trust_remote_code=True,
                    use_fast=False  # Use slow tokenizer for better compatibility
                )
                print(f"SuperBPE-t80k tokenizer loaded successfully! Vocabulary size: {len(tokenizer)}")
                print(f"Efficiency: 7.184 chars/token (vs 4.488 for GPT-2)")
            except Exception as e:
                print(f"Warning: Could not load SuperBPE-t80k tokenizer: {e}")
                print("Trying SuperBPE-t180k as fallback...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        "UW/OLMo2-8B-SuperBPE-t180k",  # Fallback to t=180k
                        token=os.environ.get("HF_TOKEN"),  # Use environment variable for security
                        trust_remote_code=True,
                        use_fast=False
                    )
                    print(f"SuperBPE-t180k loaded as fallback (31% reduction)")
                except:
                    print("Falling back to GPT-2 tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        # Use specified tokenizer (e.g., GPT-2 for compatibility)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set max length dynamically
    # Default to 4096 (model's max_position_embeddings) but allow override
    if max_length is not None:
        tokenizer.model_max_length = max_length
        if max_length > 4096:
            print(f"WARNING: Setting tokenizer max_length to {max_length}, which exceeds model's positional embedding limit (4096)")
            print("         Ensure your model has RoPE or can handle longer sequences!")
    else:
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