"""
Packed Sequence Dataset for Efficient Pre-training
Concatenates documents into continuous sequences without padding waste.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer
import numpy as np
from typing import Iterator, Dict, Optional, List
from pathlib import Path
import os
from datasets import load_dataset

class PackedPretrainDataset(IterableDataset):
    """Streaming dataset that packs multiple documents into fixed-length sequences.
    
    This eliminates padding waste by concatenating tokenized documents and
    emitting contiguous chunks of max_length tokens.
    """
    
    def __init__(
        self,
        dataset_path: str = "./experimental-pretrain-1b/dataset_1b.parquet",
        tokenizer=None,
        max_length: int = 2048,
        pack_buffer_size: int = 100000,  # Size of token buffer for packing
        add_eos_between_docs: bool = True,
        split_range: tuple = None,  # e.g., (0, 0.9) for first 90%
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_buffer_size = pack_buffer_size
        self.add_eos_between_docs = add_eos_between_docs
        
        print(f"Loading PACKED dataset from: {dataset_path}")
        
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
        
        # Load dataset
        print("Loading parquet file...")
        self.dataset = load_dataset("parquet", data_files=parquet_path, split="train")
        
        # Apply split if specified
        if split_range:
            start_idx = int(len(self.dataset) * split_range[0])
            end_idx = int(len(self.dataset) * split_range[1])
            self.dataset = self.dataset.select(range(start_idx, end_idx))
            print(f"Using samples {start_idx} to {end_idx} ({end_idx - start_idx} total)")
        
        print(f"Dataset loaded: {len(self.dataset)} examples")
        
        # Calculate total tokens (approximate)
        if 'num_tokens' in self.dataset.features:
            self.total_tokens = sum(self.dataset['num_tokens'])
            print(f"Total pre-computed tokens: {self.total_tokens:,}")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield packed sequences of exactly max_length tokens."""
        
        # Token buffer for packing
        token_buffer = []
        
        # Shuffle dataset indices for each epoch
        indices = np.random.permutation(len(self.dataset))
        
        for idx in indices:
            # Get document text
            text = self.dataset[int(idx)]['text']
            
            # Tokenize without padding or truncation
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Add EOS token between documents if specified
            if self.add_eos_between_docs and self.tokenizer.eos_token_id is not None:
                tokens.append(self.tokenizer.eos_token_id)
            
            # Add to buffer
            token_buffer.extend(tokens)
            
            # Emit full sequences when buffer is large enough
            while len(token_buffer) >= self.max_length:
                # Extract a full sequence
                sequence = token_buffer[:self.max_length]
                token_buffer = token_buffer[self.max_length:]
                
                # Convert to tensors
                input_ids = torch.tensor(sequence, dtype=torch.long)
                
                # Create labels (shift by 1 for next-token prediction)
                # We'll mask cross-document boundaries if needed
                labels = input_ids.clone()
                
                # Create attention mask (all ones for packed sequences)
                attention_mask = torch.ones_like(input_ids)
                
                yield {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
        
        # Handle remaining tokens in buffer (partial sequence)
        if token_buffer and len(token_buffer) > self.max_length // 2:
            # Only yield if we have at least half a sequence
            # Pad the rest
            pad_length = self.max_length - len(token_buffer)
            sequence = token_buffer + [self.tokenizer.pad_token_id] * pad_length
            
            input_ids = torch.tensor(sequence, dtype=torch.long)
            labels = input_ids.clone()
            labels[len(token_buffer):] = -100  # Mask padding in labels
            
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask[:len(token_buffer)] = 1
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }


class PackedPretrainDatasetWithBoundaries(PackedPretrainDataset):
    """Enhanced packed dataset that marks document boundaries in labels.
    
    This version masks tokens at document boundaries with -100 to prevent
    the model from learning to predict across unrelated documents.
    """
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield packed sequences with document boundary masking."""
        
        # Token buffer and boundary tracking
        token_buffer = []
        boundary_buffer = []  # Track where document boundaries are
        
        # Shuffle dataset indices for each epoch
        indices = np.random.permutation(len(self.dataset))
        
        for idx in indices:
            # Get document text
            text = self.dataset[int(idx)]['text']
            
            # Tokenize without padding or truncation
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Mark document boundaries
            doc_boundaries = [False] * len(tokens)
            
            # Add EOS token and mark boundary
            if self.add_eos_between_docs and self.tokenizer.eos_token_id is not None:
                tokens.append(self.tokenizer.eos_token_id)
                doc_boundaries.append(True)  # Mark EOS as boundary
            
            # Add to buffers
            token_buffer.extend(tokens)
            boundary_buffer.extend(doc_boundaries)
            
            # Emit full sequences when buffer is large enough
            while len(token_buffer) >= self.max_length:
                # Extract a full sequence
                sequence = token_buffer[:self.max_length]
                boundaries = boundary_buffer[:self.max_length]
                
                token_buffer = token_buffer[self.max_length:]
                boundary_buffer = boundary_buffer[self.max_length:]
                
                # Convert to tensors
                input_ids = torch.tensor(sequence, dtype=torch.long)
                labels = input_ids.clone()
                
                # Mask document boundaries in labels
                # This prevents learning to predict across documents
                for i, is_boundary in enumerate(boundaries):
                    if is_boundary:
                        labels[i] = -100
                
                # Create attention mask (all ones for packed sequences)
                attention_mask = torch.ones_like(input_ids)
                
                yield {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
        
        # Handle remaining tokens
        if token_buffer and len(token_buffer) > self.max_length // 2:
            pad_length = self.max_length - len(token_buffer)
            sequence = token_buffer + [self.tokenizer.pad_token_id] * pad_length
            boundaries = boundary_buffer + [True] * pad_length  # Padding is boundary
            
            input_ids = torch.tensor(sequence, dtype=torch.long)
            labels = input_ids.clone()
            
            # Mask boundaries and padding
            for i, is_boundary in enumerate(boundaries):
                if is_boundary or i >= len(token_buffer):
                    labels[i] = -100
            
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask[:len(token_buffer)] = 1
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }


def create_packed_dataloader(
    dataset_name: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    num_workers: int = 2,
    split: str = "train",
    use_boundaries: bool = True,
    pack_buffer_size: int = 100000
):
    """Create a DataLoader with packed sequences for efficient training."""
    
    # Determine split range
    if "train" in split:
        if "[" in split and "]" in split:
            # Parse split like "train[:95%]"
            import re
            match = re.search(r'\[([^:]*):([^\]]*)\]', split)
            if match:
                start = float(match.group(1).strip('%')) / 100 if match.group(1) else 0
                end = float(match.group(2).strip('%')) / 100 if match.group(2) else 1
                split_range = (start, end)
            else:
                split_range = None
        else:
            split_range = None
    else:
        split_range = None
    
    # Create appropriate dataset
    if use_boundaries:
        dataset = PackedPretrainDatasetWithBoundaries(
            dataset_path=f"./{dataset_name}/dataset_1b.parquet",
            tokenizer=tokenizer,
            max_length=max_length,
            pack_buffer_size=pack_buffer_size,
            split_range=split_range
        )
    else:
        dataset = PackedPretrainDataset(
            dataset_path=f"./{dataset_name}/dataset_1b.parquet",
            tokenizer=tokenizer,
            max_length=max_length,
            pack_buffer_size=pack_buffer_size,
            split_range=split_range
        )
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Must be 0 for IterableDataset
        pin_memory=True
    )
    
    return dataloader, dataset


# Export for use in data_pipeline
__all__ = ['PackedPretrainDataset', 'PackedPretrainDatasetWithBoundaries', 'create_packed_dataloader']