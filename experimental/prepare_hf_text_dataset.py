#!/usr/bin/env python3
"""
Prepare an HF text dataset into memory-mapped train.bin / val.bin compatible
with experimental/train.py. Defaults to skeskinen/TinyStories-hf.

- Auto-detects text column (tries: text, story, content, etc.)
- Auto-detects val split or creates one via train_test_split
- Uses tiktoken GPT-2 tokenizer by default (uint16 tokens)
- Writes metadata.json with vocab_size and token counts
"""

from __future__ import annotations

import os
import math
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from tqdm import tqdm


def get_tokenizer(name: str = "gpt2"):
    import tiktoken
    if name != "gpt2":
        raise ValueError("Only tiktoken gpt2 is supported for now")
    return tiktoken.get_encoding("gpt2")


def detect_text_column(example) -> Optional[str]:
    # Prefer common names
    preferred = ["text", "story", "content", "document", "input", "output"]
    if hasattr(example, "keys"):
        keys = list(example.keys())
        for k in preferred:
            if k in keys:
                return k
        # Fallback: pick first string-like field
        for k in keys:
            if isinstance(example[k], str):
                return k
    return None


class TokenMemmapWriter:
    def __init__(self, path: Path, dtype: np.dtype = np.uint16, initial: int = 1_000_000):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.size = max(1, int(initial))
        self.mm = np.memmap(self.path, dtype=self.dtype, mode='w+', shape=(self.size,))
        self.idx = 0

    def _ensure(self, add: int):
        need = self.idx + add
        if need <= self.size:
            return
        new_size = max(need, int(math.ceil(self.size * 1.6)))
        self.mm.flush()
        self.mm._mmap.resize(new_size * self.itemsize)
        self.mm = np.memmap(self.path, dtype=self.dtype, mode='r+', shape=(new_size,))
        self.size = new_size

    def write(self, arr: np.ndarray):
        n = int(arr.shape[0])
        if n == 0:
            return
        self._ensure(n)
        self.mm[self.idx:self.idx + n] = arr
        self.idx += n

    def close(self):
        self.mm.flush()
        # Trim file to actual length in BYTES
        self.mm._mmap.resize(self.idx * self.itemsize)
        # Reopen to validate
        self.mm = np.memmap(self.path, dtype=self.dtype, mode='r+', shape=(self.idx,))
        self.mm.flush()
        return self.idx


def tokenize_iter(dataset, text_col: str, enc, eot_token: int) -> Iterable[np.ndarray]:
    # Yield numpy arrays of token ids per example
    for ex in dataset:
        try:
            text = ex[text_col]
        except Exception:
            # Try dict-like access fallback
            text = ex.get(text_col, "") if hasattr(ex, "get") else ""
        if not isinstance(text, str) or not text:
            continue
        ids = enc.encode_ordinary(text)
        if not ids:
            continue
        ids.append(eot_token)
        yield np.asarray(ids, dtype=np.int64)  # int64 temp, cast on write


def prepare_split(dataset, out_path: Path, text_col: str, vocab_size: int, enc) -> int:
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    writer = TokenMemmapWriter(out_path, dtype=dtype, initial=2_000_000)
    count = 0
    for ids in tqdm(tokenize_iter(dataset, text_col, enc, enc.eot_token), desc=f"Writing {out_path.name}"):
        writer.write(ids.astype(dtype, copy=False))
        count += 1
    total = writer.close()
    return int(total)


def main():
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', type=str, default=os.getenv('HF_DATASET', 'skeskinen/TinyStories-hf'))
    parser.add_argument('--train-split', type=str, default=None, help='Name of train split (auto if None)')
    parser.add_argument('--val-split', type=str, default=None, help='Name of validation split (auto if None)')
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--val-ratio', type=float, default=0.02, help='If no val split, carve from train')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    enc = get_tokenizer(args.tokenizer)

    print(f"Loading dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id)

    # Determine splits
    split_names = list(getattr(ds, 'keys', lambda: [])())
    train_split = args.train_split or ("train" if "train" in split_names else (split_names[0] if split_names else None))
    val_split = args.val_split
    if not val_split:
        for cand in ("validation", "valid", "val", "test"):
            if cand in split_names and cand != train_split:
                val_split = cand
                break

    # If no val split present, create from train
    if val_split is None:
        print("No validation split found; creating one via train_test_split...")
        tmp = ds[train_split].train_test_split(test_size=args.val_ratio, seed=42)
        ds = {"train": tmp["train"], "val": tmp["test"]}
        train_key, val_key = "train", "val"
    else:
        train_key, val_key = train_split, val_split

    # Detect text column
    example = next(iter(ds[train_key])) if hasattr(ds[train_key], '__iter__') else ds[train_key][0]
    text_col = detect_text_column(example)
    if not text_col:
        raise RuntimeError("Could not detect a text column in the dataset")
    print(f"Using text column: {text_col}")

    vocab_size = enc.n_vocab

    # Write train
    train_tokens = prepare_split(ds[train_key], out_dir / 'train.bin', text_col, vocab_size, enc)
    # Write val
    val_tokens = prepare_split(ds[val_key], out_dir / 'val.bin', text_col, vocab_size, enc)

    # Metadata
    meta = {
        'vocab_size': vocab_size,
        'tokenizer': f'tiktoken-{args.tokenizer}',
        'train_tokens': int(train_tokens),
        'val_tokens': int(val_tokens),
        'dataset_id': args.dataset_id,
        'train_split': train_key,
        'val_split': val_key,
        'text_column': text_col,
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens: {val_tokens:,}")
    print(f"Total tokens: {train_tokens + val_tokens:,}")


if __name__ == '__main__':
    main()

