#!/usr/bin/env python3
"""
SFT Data Preparation Script for Parquet Format
Converts parquet conversational format to training data for the elite 2.218 model.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from transformers import AutoTokenizer
from typing import List, Dict, Any
import os

def load_tokenizer():
    """Load the same tokenizer used for pretraining."""
    print("Loading SmolLM2-135M tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        use_fast=True
    )
    print(f"  Vocab size: {len(tokenizer)}")
    return tokenizer

def format_conversation(messages: List[Dict[str, str]], tokenizer) -> str:
    """
    Convert messages to training format.
    Format: User: {content}\nAssistant: {content}\n
    """
    formatted_text = ""
    
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        
        if role == "user":
            formatted_text += f"User: {content}\n"
        elif role == "assistant":
            formatted_text += f"Assistant: {content}\n"
        else:
            # Handle system messages or others
            formatted_text += f"{role.title()}: {content}\n"
    
    return formatted_text

def create_sft_training_example(messages: List[Dict[str, str]], tokenizer, max_length: int = 2048):
    """
    Create a training example from conversation messages.
    
    Key insight for SFT:
    - We predict ONLY the assistant responses
    - User messages provide context but aren't predicted
    - This teaches instruction following
    """
    formatted_text = format_conversation(messages, tokenizer)
    
    # Tokenize the full conversation
    tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
    
    # Truncate if too long
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # Create labels - mask user inputs, predict only assistant responses
    labels = []
    current_text = ""
    
    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()
        
        if role == "user":
            user_text = f"User: {content}\n"
            user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
            # Mask user tokens (don't predict them)
            labels.extend([-100] * len(user_tokens))
            current_text += user_text
            
        elif role == "assistant":
            assistant_text = f"Assistant: {content}\n"
            assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
            # Predict assistant tokens
            labels.extend(assistant_tokens)
            current_text += assistant_text
    
    # Ensure lengths match
    if len(labels) > len(tokens):
        labels = labels[:len(tokens)]
    elif len(labels) < len(tokens):
        # Pad labels if needed
        labels.extend([-100] * (len(tokens) - len(labels)))
    
    return {
        "input_ids": tokens,
        "labels": labels,
        "length": len(tokens),
        "conversation_turns": len([m for m in messages if m["role"] == "assistant"])
    }

def process_parquet_dataset(input_file: str, output_dir: str, tokenizer, max_length: int = 2048):
    """Process the parquet SFT dataset and create training files."""
    print(f"Processing parquet SFT dataset: {input_file}")
    
    # Load parquet data
    print("Loading parquet file...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Show column info
    print(f"Columns: {list(df.columns)}")
    
    # Determine format - look for 'messages' column or conversation format
    if 'messages' in df.columns:
        print("Found 'messages' column - assuming standard format")
        message_column = 'messages'
    elif 'conversation' in df.columns:
        print("Found 'conversation' column")
        message_column = 'conversation'
    else:
        print("Available columns:", df.columns.tolist())
        # Try to find likely column
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['message', 'chat', 'conversation', 'dialog']):
                message_column = col
                print(f"Using column: {message_column}")
                break
        else:
            raise ValueError("Could not find messages column. Available columns: " + str(df.columns.tolist()))
    
    # Process conversations
    processed_examples = []
    total_tokens = 0
    skipped = 0
    
    for i, row in df.iterrows():
        try:
            # Extract messages
            messages = row[message_column]
            
            # Handle different formats
            if isinstance(messages, str):
                # Parse JSON string
                messages = json.loads(messages)
            elif isinstance(messages, list):
                # Already a list
                pass
            else:
                print(f"  Unknown message format at row {i}: {type(messages)}")
                skipped += 1
                continue
            
            # Validate messages format
            if not isinstance(messages, list) or len(messages) < 2:
                skipped += 1
                continue
            
            # Check if messages have required fields
            if not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in messages):
                skipped += 1
                continue
            
            example = create_sft_training_example(messages, tokenizer, max_length)
            
            # Filter out very short examples (< 20 tokens)
            if example["length"] < 20:
                skipped += 1
                continue
                
            processed_examples.append(example)
            total_tokens += example["length"]
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{len(df)} conversations")
                
        except Exception as e:
            print(f"  Error processing conversation {i}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"  Valid examples: {len(processed_examples)}")
    print(f"  Skipped examples: {skipped}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens per example: {total_tokens/len(processed_examples):.1f}")
    
    # Split into train/val (90/10)
    np.random.seed(42)  # Reproducible split
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
        # Collect all tokens
        all_tokens = []
        all_labels = []
        
        for example in examples:
            all_tokens.extend(example["input_ids"])
            all_labels.extend(example["labels"])
        
        # Convert to numpy arrays
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        labels_array = np.array(all_labels, dtype=np.int32)  # int32 for -100 values
        
        # Save as binary files
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
        "source_format": "parquet"
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSFT dataset prepared successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  Ready for training with train_sft.py")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Prepare SFT training data from parquet')
    parser.add_argument('input_file', type=str, help='Input parquet file with conversations')
    parser.add_argument('--output_dir', type=str, default='data_sft', help='Output directory')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Process dataset
    metadata = process_parquet_dataset(
        args.input_file,
        args.output_dir,
        tokenizer,
        args.max_length
    )
    
    print("\n" + "="*60)
    print("SFT DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_dir}")
    print(f"Train tokens: {metadata['train_tokens']:,}")
    print(f"Val tokens: {metadata['val_tokens']:,}")
    print(f"Train examples: {metadata['train_examples']:,}")
    print(f"\nNext step:")
    print(f"python train_sft.py --data_dir {args.output_dir} --base_model best_model_fp8_optimized.pt")
    print("="*60)

if __name__ == "__main__":
    main()