#!/usr/bin/env python3
"""
Fast SFT Data Preparation Script for Parquet Format
Uses batch tokenization for much faster processing.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from transformers import AutoTokenizer
from typing import List, Dict, Any
import os
from tqdm import tqdm

def load_tokenizer():
    """Load the same tokenizer used for pretraining."""
    print("Loading SmolLM2-135M tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        use_fast=True  # Already fast!
    )
    print(f"  Vocab size: {len(tokenizer)}")
    return tokenizer

def format_conversations_batch(messages_batch: List[List[Dict[str, str]]]) -> List[str]:
    """
    Convert multiple conversations to training format.
    Format: User: {content}\nAssistant: {content}\n
    """
    formatted_texts = []
    
    for messages in messages_batch:
        formatted_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"].strip()
            
            if role == "user":
                formatted_text += f"User: {content}\n"
            elif role == "assistant":
                formatted_text += f"Assistant: {content}\n"
            else:
                formatted_text += f"{role.title()}: {content}\n"
        
        formatted_texts.append(formatted_text)
    
    return formatted_texts

def create_sft_training_examples_batch(
    messages_batch: List[List[Dict[str, str]]], 
    tokenizer, 
    max_length: int = 2048
):
    """
    Create training examples from a batch of conversations.
    Uses batch tokenization for speed.
    """
    # Format all conversations
    formatted_texts = format_conversations_batch(messages_batch)
    
    # Batch tokenize all conversations at once
    batch_encoding = tokenizer(
        formatted_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False
    )
    
    results = []
    
    # Process each conversation to create labels
    for idx, (messages, input_ids) in enumerate(zip(messages_batch, batch_encoding['input_ids'])):
        # Create labels with masking
        labels = []
        
        # Tokenize each part to figure out where to mask
        for message in messages:
            role = message["role"]
            content = message["content"].strip()
            
            if role == "user":
                user_text = f"User: {content}\n"
                user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
                labels.extend([-100] * len(user_tokens))
            elif role == "assistant":
                assistant_text = f"Assistant: {content}\n"
                assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
                labels.extend(assistant_tokens)
        
        # Ensure lengths match
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        
        results.append({
            "input_ids": input_ids,
            "labels": labels,
            "length": len(input_ids),
            "conversation_turns": len([m for m in messages if m["role"] == "assistant"])
        })
    
    return results

def process_parquet_dataset_fast(
    input_file: str, 
    output_dir: str, 
    tokenizer, 
    max_length: int = 2048,
    max_examples: int = None,
    sample_rate: float = 1.0,
    batch_size: int = 100
):
    """Process the parquet SFT dataset with batch tokenization."""
    print(f"Processing parquet SFT dataset: {input_file}")
    
    # Load parquet data
    print("Loading parquet file...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Sample if requested
    if sample_rate < 1.0:
        n_samples = int(len(df) * sample_rate)
        df = df.sample(n=n_samples, random_state=42)
        print(f"Sampled {len(df)} rows ({sample_rate*100:.1f}%)")
    
    if max_examples and len(df) > max_examples:
        df = df.head(max_examples)
        print(f"Limited to {max_examples} examples")
    
    # Show column info
    print(f"Columns: {list(df.columns)}")
    
    # Determine format
    if 'messages' in df.columns:
        print("Found 'messages' column - assuming standard format")
        message_column = 'messages'
    elif 'conversation' in df.columns:
        print("Found 'conversation' column")
        message_column = 'conversation'
    else:
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['message', 'chat', 'conversation', 'dialog']):
                message_column = col
                print(f"Using column: {message_column}")
                break
        else:
            raise ValueError("Could not find messages column. Available columns: " + str(df.columns.tolist()))
    
    # Process in batches
    processed_examples = []
    total_tokens = 0
    skipped = 0
    
    print(f"Processing {len(df)} conversations in batches of {batch_size}...")
    
    # Collect valid conversations in batches
    batch_messages = []
    batch_indices = []
    
    with tqdm(total=len(df), desc="Processing") as pbar:
        for i, row in df.iterrows():
            try:
                # Extract messages
                messages = row[message_column]
                
                # Handle different formats
                if isinstance(messages, str):
                    messages = json.loads(messages)
                elif isinstance(messages, np.ndarray):
                    messages = messages.tolist()
                elif not isinstance(messages, list):
                    skipped += 1
                    pbar.update(1)
                    continue
                
                # Validate
                if len(messages) < 2:
                    skipped += 1
                    pbar.update(1)
                    continue
                
                if not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in messages):
                    skipped += 1
                    pbar.update(1)
                    continue
                
                batch_messages.append(messages)
                batch_indices.append(i)
                
                # Process batch when full
                if len(batch_messages) >= batch_size:
                    batch_examples = create_sft_training_examples_batch(
                        batch_messages, tokenizer, max_length
                    )
                    
                    for example in batch_examples:
                        if example["length"] >= 20:  # Filter short examples
                            processed_examples.append(example)
                            total_tokens += example["length"]
                        else:
                            skipped += 1
                    
                    batch_messages = []
                    batch_indices = []
                
                pbar.update(1)
                
            except Exception as e:
                if i < 5:  # Only print first few errors
                    print(f"  Error processing conversation {i}: {e}")
                skipped += 1
                pbar.update(1)
                continue
    
    # Process remaining batch
    if batch_messages:
        batch_examples = create_sft_training_examples_batch(
            batch_messages, tokenizer, max_length
        )
        for example in batch_examples:
            if example["length"] >= 20:
                processed_examples.append(example)
                total_tokens += example["length"]
            else:
                skipped += 1
    
    print(f"\nProcessing complete!")
    print(f"  Valid examples: {len(processed_examples)}")
    print(f"  Skipped examples: {skipped}")
    print(f"  Total tokens: {total_tokens:,}")
    if processed_examples:
        print(f"  Average tokens per example: {total_tokens/len(processed_examples):.1f}")
    
    # Split into train/val (90/10)
    np.random.seed(42)
    indices = np.random.permutation(len(processed_examples))
    split_idx = int(0.9 * len(processed_examples))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_examples = [processed_examples[i] for i in train_indices]
    val_examples = [processed_examples[i] for i in val_indices]
    
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Val examples: {len(val_examples)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save training examples
    def save_examples(examples, split_name):
        all_tokens = []
        all_labels = []
        
        for example in examples:
            all_tokens.extend(example["input_ids"])
            all_labels.extend(example["labels"])
        
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        labels_array = np.array(all_labels, dtype=np.int32)
        
        tokens_file = output_path / f"{split_name}_tokens.bin"
        labels_file = output_path / f"{split_name}_labels.bin"
        
        tokens_array.tofile(tokens_file)
        labels_array.tofile(labels_file)
        
        print(f"  Saved {split_name}: {len(all_tokens):,} tokens")
        return len(all_tokens)
    
    train_tokens = save_examples(train_examples, "train")
    val_tokens = save_examples(val_examples, "val")
    
    # Save metadata
    metadata = {
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
        "max_length": max_length,
        "total_examples": len(processed_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "vocab_size": len(tokenizer),
        "format": "sft_conversational",
        "source_file": input_file,
        "source_format": "parquet",
        "batch_tokenization": True
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSFT dataset prepared successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  Ready for training with train_sft.py")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Fast SFT data preparation with batch tokenization')
    parser.add_argument('input_file', type=str, help='Input parquet file with conversations')
    parser.add_argument('--output_dir', type=str, default='data_sft', help='Output directory')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--max_examples', type=int, default=None, help='Maximum examples to process')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='Fraction of data to sample (0.1 = 10%)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for tokenization')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Process dataset
    metadata = process_parquet_dataset_fast(
        args.input_file,
        args.output_dir,
        tokenizer,
        args.max_length,
        args.max_examples,
        args.sample_rate,
        args.batch_size
    )
    
    print("\n" + "="*60)
    print("SFT DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_dir}")
    if metadata:
        print(f"Train tokens: {metadata['train_tokens']:,}")
        print(f"Val tokens: {metadata['val_tokens']:,}")
        print(f"Train examples: {metadata['train_examples']:,}")
    print(f"\nNext step:")
    print(f"python train_sft.py --data_dir {args.output_dir} --base_model yxanul-base/best_model_fp8_optimized.pt")
    print("="*60)

if __name__ == "__main__":
    main()