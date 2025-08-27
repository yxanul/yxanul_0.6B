"""
Data Pipeline for Yxanul 0.6B Training
Implements efficient streaming data loading from HuggingFace datasets.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, Optional, Iterator
import random
from pathlib import Path
import os  # Required for environment variables

# Maximum buffer size to prevent memory leaks
MAX_BUFFER_SIZE = 100000  # Prevent OOM with very long documents


class PretrainDataset(Dataset):
    """Non-streaming dataset for pre-downloaded datasets with proper indexing.
    
    Optimized for Yxanul/experimental-pretrain-1b and similar pre-downloaded datasets.
    Supports:
    - Random access with __getitem__ for efficient DataLoader operations
    - Multiple workers with proper memory pinning
    - Efficient shuffling and batching (dataset already shuffled with seed=42)
    - Pre-tokenization caching for faster training
    - NO OVERLAP - maximizes efficiency with 1B tokens
    
    Expected format for Yxanul dataset:
    - text (string): The text content for training
    - source (string): Source category - one of ['fineweb', 'math', 'code']
    - num_tokens (int64): Pre-computed token count
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_length: int = 2048,
        split: str = "train",
        add_eos_between_docs: bool = True,
        cache_dir: str = None,
        num_proc: int = 4
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.add_eos_between_docs = add_eos_between_docs
        
        # No overlap - use full sequence length as stride for maximum efficiency
        self.stride = max_length
        
        print(f"Initializing PretrainDataset with seq_len={max_length} (no overlap)")
        
        # Load dataset - handle both HF hub and local parquet file
        print(f"Loading dataset: {dataset_name}")
        
        # Check if it's a local parquet file first
        import os
        from pathlib import Path
        
        # Check for local dataset_1b.parquet file
        local_paths = [
            Path("dataset_1b.parquet"),
            Path("experimental-pretrain-1b/dataset_1b.parquet"),
            Path(f"{dataset_name}/dataset_1b.parquet") if "/" not in dataset_name else None,
            Path(cache_dir) / "dataset_1b.parquet" if cache_dir else None,
        ]
        
        dataset_loaded = False
        for local_path in local_paths:
            if local_path and local_path.exists():
                print(f"Found local dataset file: {local_path}")
                self.dataset = load_dataset(
                    "parquet",
                    data_files=str(local_path),
                    split=split,
                    num_proc=num_proc
                )
                dataset_loaded = True
                break
        
        # If not found locally, try loading from HuggingFace
        if not dataset_loaded:
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                num_proc=num_proc
            )
        
        print(f"Dataset loaded: {len(self.dataset)} examples")
        if hasattr(self.dataset, 'features'):
            print(f"Dataset features: {list(self.dataset.features.keys())}")
        
        # Pre-tokenize and create chunks
        self._prepare_chunks()
    
    def _prepare_chunks(self):
        """Pre-tokenize all texts and create fixed-length chunks (no overlap)."""
        print("Pre-tokenizing dataset and creating non-overlapping chunks...")
        self.chunks = []
        buffer = []
        
        # Process in batches for efficiency
        batch_size = 1000
        total_docs = len(self.dataset)
        
        for idx in range(0, total_docs, batch_size):
            batch_end = min(idx + batch_size, total_docs)
            batch = self.dataset[idx:batch_end]
            
            # Process each document in batch
            texts = batch['text'] if isinstance(batch['text'], list) else [batch['text']]
            
            for text in texts:
                if not text or len(text) < 100:  # Skip very short texts
                    continue
                
                # Tokenize
                tokens = self.tokenizer.encode(
                    text,
                    truncation=False,
                    add_special_tokens=True
                )
                
                # Add to buffer with EOS token
                buffer.extend(tokens)
                if self.add_eos_between_docs and self.tokenizer.eos_token_id is not None:
                    buffer.append(self.tokenizer.eos_token_id)
                
                # Create non-overlapping chunks from buffer
                while len(buffer) >= self.max_length:
                    chunk = buffer[:self.max_length]
                    buffer = buffer[self.max_length:]  # No overlap - move by full length
                    
                    # Create input_ids and labels
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    labels = input_ids.clone()
                    labels[0] = -100  # Mask first position
                    
                    self.chunks.append({
                        'input_ids': input_ids,
                        'labels': labels,
                        'attention_mask': torch.ones(self.max_length, dtype=torch.long)
                    })
            
            # Progress update
            if (idx + batch_size) % 10000 == 0:
                print(f"  Processed {min(idx + batch_size, total_docs)}/{total_docs} documents, "
                      f"created {len(self.chunks)} chunks")
        
        print(f"Preprocessing complete: {len(self.chunks)} chunks created from {total_docs} documents")
        print(f"Tokens per chunk: {self.max_length} (no overlap for maximum efficiency)")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]


class StreamingDataset(IterableDataset):
    """Dataset for efficient data loading from pre-downloaded datasets.
    
    This dataset properly handles document boundaries by inserting EOS tokens between
    documents, which is crucial for language model training to learn when to stop
    generating and to avoid treating concatenated documents as continuous text.
    
    Supports both streaming (for very large datasets) and pre-downloaded datasets.
    For Yxanul/experimental-pretrain-1b, expects format:
    - text (string): The text content for training
    - source (string): Source category - one of ['fineweb', 'math', 'code'] 
    - num_tokens (int64): Pre-computed token count
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_length: int = 2048,
        stride: int = None,  # Will be computed based on current sequence length
        stage_config: Dict = None,
        split: str = "train",
        add_eos_between_docs: bool = True,
        streaming: bool = False  # Default to non-streaming for better performance
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stage_config = stage_config if stage_config is not None else {}
        self.split = split
        self.add_eos_between_docs = add_eos_between_docs  # Add EOS tokens at document boundaries
        self.streaming = streaming  # Whether to stream or load fully
        
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
            
            # Special handling for Yxanul/experimental-pretrain-1b dataset
            if "yxanul" in self.dataset_name.lower() or "experimental-pretrain-1b" in self.dataset_name.lower():
                print(f"Loading Yxanul experimental-pretrain-1b dataset from HuggingFace...")
                print("Dataset format: text, source, num_tokens fields")
                
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
                    # Load from HuggingFace (with special handling for Yxanul dataset)
                    if "yxanul" in self.dataset_name.lower() or "experimental-pretrain" in self.dataset_name.lower():
                        # Load the Yxanul experimental dataset
                        print("Loading Yxanul/experimental-pretrain-1b dataset...")
                        self.dataset = load_dataset(
                            self.dataset_name,
                            split=self.split,
                            streaming=self.streaming,  # Use configured streaming setting
                            num_proc=4 if not self.streaming else 1  # Multi-proc for non-streaming
                        )
                        if not self.streaming:
                            print(f"Dataset loaded: {len(self.dataset)} examples")
                            # Log dataset statistics
                            if hasattr(self.dataset, 'features'):
                                print(f"Dataset features: {list(self.dataset.features.keys())}")
                    else:
                        # Regular HuggingFace dataset
                        self.dataset = load_dataset(
                            self.dataset_name,
                            split=self.split,
                            streaming=self.streaming,
                            num_proc=4 if not self.streaming else 1
                        )
                        if not self.streaming:
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
    add_eos_between_docs: bool = True,
    streaming: bool = False,
    cache_dir: str = "./cache"
) -> DataLoader:
    """Create a DataLoader for either streaming or pre-downloaded dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'Yxanul/experimental-pretrain-1b')
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        stage_config: Configuration for curriculum stages
        num_workers: Number of dataloader workers
        split: Dataset split to use
        add_eos_between_docs: Whether to add EOS tokens between documents
        streaming: Whether to stream dataset (False for pre-downloaded)
        cache_dir: Cache directory for downloaded datasets
    """
    
    # Use PretrainDataset for Yxanul experimental dataset or when explicitly non-streaming
    if (not streaming and ("yxanul" in dataset_name.lower() or "experimental-pretrain" in dataset_name.lower())):
        print(f"Using PretrainDataset for optimal performance with {dataset_name}")
        dataset = PretrainDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=max_length,
            split=split,
            add_eos_between_docs=add_eos_between_docs,
            cache_dir=cache_dir,
            num_proc=4  # Use multiple processes for preprocessing
        )
        # Dataset is already shuffled with seed=42, but DataLoader can shuffle indices
        shuffle = ("train" in split)
    else:
        # Use StreamingDataset for other datasets or when streaming
        dataset = StreamingDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=None,  # Will be computed based on current sequence length
            stage_config=stage_config,
            split=split,
            add_eos_between_docs=add_eos_between_docs,
            streaming=streaming
        )
        # StreamingDataset handles its own shuffling
        shuffle = False
    
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
    
    # Build DataLoader kwargs conditionally for better compatibility
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True  # Drop incomplete batches to maintain FP8 requirements
    }
    
    # Add shuffle for non-streaming datasets
    if isinstance(dataset, PretrainDataset):
        dataloader_kwargs['shuffle'] = shuffle
    
    # Only add worker-specific args when using workers
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
        dataloader_kwargs['prefetch_factor'] = 4
    else:
        dataloader_kwargs['persistent_workers'] = False
        # Don't set prefetch_factor when num_workers is 0
    
    dataloader = DataLoader(**dataloader_kwargs)
    
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
        
        # Try loading from local cache first (FAST tokenizers)
        cache_paths = ["./tokenizer_cache/superbpe-t80k-fast", "./tokenizer_cache/superbpe-t180k-fast"]
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
                    use_fast=True  # Use FAST tokenizer for 35% fewer tokens!
                )
                print(f"SuperBPE-t80k FAST tokenizer loaded successfully! Vocabulary size: {len(tokenizer)}")
                print(f"TRUE Efficiency: ~667M tokens for 1B dataset (35% compression!)")
            except Exception as e:
                print(f"Warning: Could not load SuperBPE-t80k tokenizer: {e}")
                print("Trying SuperBPE-t180k as fallback...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        "UW/OLMo2-8B-SuperBPE-t180k",  # Fallback to t=180k
                        token=os.environ.get("HF_TOKEN"),  # Use environment variable for security
                        trust_remote_code=True,
                        use_fast=True  # Use FAST tokenizer here too!
                    )
                    print(f"SuperBPE-t180k FAST loaded as fallback (still efficient!)")
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


class CurriculumStreamingDataset(IterableDataset):
    """Curriculum-aware streaming dataset that streams directly from HuggingFace.
    
    This dataset:
    1. Streams directly from HuggingFace without pre-downloading
    2. Mixes multiple datasets (FineWeb, math, code) with evolving ratios
    3. Adjusts sequence length per curriculum stage
    4. Tracks token consumption for automatic stage transitions
    """
    
    def __init__(
        self,
        curriculum_config: Dict,
        tokenizer,
        current_stage: int = 0,
        split: str = "train"
    ):
        self.curriculum_config = curriculum_config
        self.tokenizer = tokenizer
        self.current_stage = current_stage
        self.split = split
        
        # Get stages from config
        self.stages = curriculum_config.get('training', {}).get('curriculum_stages', [])
        if not self.stages:
            raise ValueError("No curriculum stages found in config")
        
        # Current stage configuration
        self.stage_config = self.stages[self.current_stage]
        self.seq_len = self.stage_config['seq_len']
        self.dataset_mix = self.stage_config['dataset_mix']
        
        # Dataset sources
        self.dataset_sources = {
            'fineweb': {
                'path': 'HuggingFaceFW/fineweb-edu',
                'config': 'sample-10BT',  # 10B token sample
                'text_column': 'text',
                'streaming': True
            },
            'math': {
                'path': 'Yxanul/cc-math-finest',
                'text_column': 'text',
                'streaming': True
            },
            'code': {
                'path': 'Yxanul/python-finest-pretrain',
                'text_column': 'text',
                'streaming': True
            }
        }
        
        # Initialize streaming datasets
        self._init_datasets()
    
    def _init_datasets(self):
        """Initialize streaming datasets from HuggingFace."""
        self.datasets = {}
        self.iterators = {}
        
        for name, config in self.dataset_sources.items():
            if self.dataset_mix.get(name, 0) > 0:
                print(f"Initializing {name} dataset from {config['path']}...")
                
                try:
                    # Load with streaming to avoid downloading entire dataset
                    dataset_args = {
                        'path': config['path'],
                        'split': self.split,
                        'streaming': config.get('streaming', True)
                    }
                    
                    # Add config name if specified (for FineWeb-Edu)
                    if 'config' in config:
                        dataset_args['name'] = config['config']
                    
                    dataset = load_dataset(**dataset_args)
                    self.datasets[name] = dataset
                    self.iterators[name] = iter(dataset)
                    print(f"  [OK] {name} dataset ready for streaming")
                    
                except Exception as e:
                    print(f"  [FAIL] Failed to load {name}: {e}")
                    # Create dummy iterator that yields empty
                    self.datasets[name] = None
                    self.iterators[name] = iter([])
    
    def update_stage(self, new_stage: int):
        """Update to a new curriculum stage."""
        if new_stage < len(self.stages):
            self.current_stage = new_stage
            self.stage_config = self.stages[new_stage]
            self.seq_len = self.stage_config['seq_len']
            self.dataset_mix = self.stage_config['dataset_mix']
            
            print(f"\nTransitioned to Stage {new_stage + 1}: {self.stage_config['name']}")
            print(f"  Sequence length: {self.seq_len}")
            print(f"  Dataset mix: {self.dataset_mix}")
            
            # Reinitialize datasets if mix changed significantly
            self._init_datasets()
    
    def __iter__(self):
        """Stream and tokenize samples based on current stage mix."""
        buffer = []
        
        # Normalize weights
        total_weight = sum(self.dataset_mix.values())
        if total_weight == 0:
            raise ValueError("All dataset weights are zero!")
        
        weights = {k: v/total_weight for k, v in self.dataset_mix.items()}
        dataset_names = list(weights.keys())
        dataset_probs = [weights[name] for name in dataset_names]
        
        while True:
            # Sample a dataset based on current mix
            dataset_name = np.random.choice(dataset_names, p=dataset_probs)
            
            # Skip if dataset not available
            if self.datasets.get(dataset_name) is None:
                continue
            
            try:
                # Get next sample from chosen dataset
                sample = next(self.iterators[dataset_name])
                
                # Get text from sample
                text_column = self.dataset_sources[dataset_name]['text_column']
                text = sample.get(text_column, '')
                
                if not text or len(text) < 100:
                    continue
                
                # Tokenize
                tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=True)
                
                # Add to buffer with EOS token between documents
                buffer.extend(tokens)
                if self.tokenizer.eos_token_id is not None:
                    buffer.append(self.tokenizer.eos_token_id)
                
                # Prevent memory overflow
                if len(buffer) > MAX_BUFFER_SIZE:
                    buffer = buffer[-MAX_BUFFER_SIZE:]
                
                # Create chunks of current sequence length
                while len(buffer) >= self.seq_len:
                    chunk = buffer[:self.seq_len]
                    # Use 50% stride for efficiency (or full sequence for very short seqs)
                    stride = self.seq_len if self.seq_len <= 64 else max(1, self.seq_len // 2)
                    buffer = buffer[stride:]
                    
                    # Create tensors
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    labels = input_ids.clone()
                    labels[0] = -100  # Mask first position
                    
                    yield {
                        'input_ids': input_ids,
                        'labels': labels,
                        'attention_mask': torch.ones(self.seq_len, dtype=torch.long),
                        'dataset_source': dataset_name  # Track source for analysis
                    }
                    
            except StopIteration:
                # Restart this dataset's iterator
                if self.datasets[dataset_name] is not None:
                    self.iterators[dataset_name] = iter(self.datasets[dataset_name])


def create_curriculum_dataloader(
    curriculum_config: Dict,
    tokenizer,
    batch_size: int,
    current_stage: int = 0,
    num_workers: int = 0,  # Streaming doesn't parallelize well
    split: str = "train"
) -> DataLoader:
    """Create a curriculum-aware streaming dataloader."""
    
    dataset = CurriculumStreamingDataset(
        curriculum_config=curriculum_config,
        tokenizer=tokenizer,
        current_stage=current_stage,
        split=split
    )
    
    # Adjust batch size to be multiple of 8 for FP8
    if batch_size % 8 != 0:
        batch_size = max(8, (batch_size // 8) * 8)
    
    # Simple collate for streaming
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=fp8_collate_fn,
        num_workers=num_workers,  # Usually 0 for streaming
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, dataset