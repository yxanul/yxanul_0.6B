#!/usr/bin/env python3
"""
Robust TinyStories preparation script.
Attempts HF datasets first; on failure, falls back to downloading parquet files
via huggingface_hub and loading them locally. Writes train.bin/val.bin and
metadata.json compatible with experimental/train.py.
"""

import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def _load_tokenizer(tokenizer_name: str = "gpt2"):
    import tiktoken
    if tokenizer_name != "gpt2":
        raise ValueError("This script currently supports tiktoken GPT-2 only.")
    return tiktoken.get_encoding("gpt2")


def _try_load_with_datasets(repo_id: str):
    try:
        from datasets import load_dataset
        print("Attempting datasets.load_dataset(...)")
        ds = load_dataset(repo_id)
        # Expect 'train' and 'validation'
        if not ("train" in ds and ("validation" in ds or "valid" in ds)):
            print("Dataset loaded but expected splits not found; continuing...")
            return None
        return ds
    except Exception as e:
        print(f"datasets.load_dataset failed: {e}")
        return None


def _download_parquets(repo_id: str, revision: str | None = None) -> Dict[str, List[str]]:
    """Download parquet files from Hub and return local file paths by split."""
    from huggingface_hub import list_repo_files, hf_hub_download

    print(f"Listing files for {repo_id}...")
    files = list_repo_files(repo_id, revision=revision)
    parquet_files = [f for f in files if f.lower().endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError("No parquet files found in repository")

    split_map = {"train": [], "validation": []}
    for f in parquet_files:
        lf = f.lower()
        if "train" in lf:
            split = "train"
        elif "valid" in lf or "val" in lf:
            split = "validation"
        else:
            # Default to train if unspecified
            split = "train"
        local_path = hf_hub_download(repo_id=repo_id, filename=f, revision=revision)
        split_map[split].append(local_path)

    if not split_map["train"]:
        raise FileNotFoundError("No training parquet files resolved from repo")
    if not split_map["validation"]:
        print("Warning: No validation parquet files found; will create empty val.bin")
    return split_map


def _load_local_parquets(split_files: Dict[str, List[str]]):
    from datasets import load_dataset
    data_files = {}
    if split_files.get("train"):
        data_files["train"] = split_files["train"]
    if split_files.get("validation"):
        data_files["validation"] = split_files["validation"]
    print("Loading local parquet files via datasets parquet builder...")
    return load_dataset("parquet", data_files=data_files)


def _write_split(tokens_list: List[int], out_path: Path, dtype=np.uint16):
    arr = np.asarray(tokens_list, dtype=dtype)
    mm = np.memmap(out_path, dtype=dtype, mode="w+", shape=(arr.shape[0],))
    mm[:] = arr
    mm.flush()


def main():
    repo_id = os.getenv("TINYSTORIES_REPO", "roneneldan/TinyStories")
    revision = os.getenv("TINYSTORIES_REVISION")  # optional
    out_dir = Path(os.getenv("OUT_DIR", "data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    enc = _load_tokenizer("gpt2")
    vocab_size = enc.n_vocab

    print("Loading TinyStories dataset...")
    ds = _try_load_with_datasets(repo_id)
    if ds is None:
        print("Falling back to direct parquet download from Hub...")
        split_files = _download_parquets(repo_id, revision)
        ds = _load_local_parquets(split_files)

    # Prepare splits mapping to expected output names
    splits = []
    if "train" in ds:
        splits.append(("train", ds["train"]))
    if "validation" in ds:
        splits.append(("val", ds["validation"]))
    elif "valid" in ds:
        splits.append(("val", ds["valid"]))
    else:
        # Create empty validation
        splits.append(("val", []))

    for out_name, data in splits:
        print(f"\nProcessing {out_name} split...")
        # Tokenize all texts; accept both dicts and HF examples
        token_ids: List[int] = []
        count = 0
        for ex in tqdm(data, desc=f"Tokenizing {out_name}"):
            if isinstance(ex, dict):
                text = ex.get("text") or ex.get("story") or ""
            else:
                try:
                    text = ex["text"]
                except Exception:
                    try:
                        text = ex["story"]
                    except Exception:
                        text = ""
            if not text:
                continue
            token_ids.extend(enc.encode_ordinary(text))
            token_ids.append(enc.eot_token)
            count += 1
        print(f"Tokenized {count} examples; total tokens: {len(token_ids):,}")

        out_file = out_dir / f"{out_name}.bin"
        print(f"Writing {out_file} ...")
        _write_split(token_ids, out_file, dtype=np.uint16 if vocab_size <= 65535 else np.uint32)

    # Write metadata.json for training script
    meta = {
        "vocab_size": vocab_size,
        "tokenizer": "tiktoken-gpt2",
        "train_tokens": int(len(np.memmap(out_dir / "train.bin", dtype=np.uint16 if vocab_size <= 65535 else np.uint32, mode="r"))),
        "val_tokens": int(len(np.memmap(out_dir / "val.bin", dtype=np.uint16 if vocab_size <= 65535 else np.uint32, mode="r"))) if (out_dir / "val.bin").exists() else 0,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDataset preparation complete!")
    print(f"Train tokens: {meta['train_tokens']:,}")
    print(f"Val tokens: {meta['val_tokens']:,}")
    print(f"Total tokens: {meta['train_tokens'] + meta['val_tokens']:,}")


if __name__ == "__main__":
    main()
