# Dataset Setup Guide

## Quick Start

1. **Clone the dataset** (one-time, ~3.6GB):
```bash
git clone https://huggingface.co/datasets/Yxanul/wikipedia-2k-high-quality ./data/wikipedia
```

2. **Prepare for training** (creates train/val splits):
```bash
python prepare_dataset.py
```

3. **Start training**:
```bash
python train.py
```

## Why Git Clone?

- **No rate limits**: Downloads as a single Git repository
- **Faster**: ~80 MB/s vs repeated API calls
- **Reliable**: No 429 errors or connection issues
- **Standard practice**: How HuggingFace intends large datasets to be downloaded

## Dataset Structure

After preparation:
- **Training examples**: ~226,000
- **Validation examples**: ~12,000  
- **Total tokens**: ~450M
- **Cached location**: `./data/processed_dataset/`

## Disk Space Requirements

- Raw dataset: 3.6GB (can be deleted after preparation)
- Processed dataset: ~1.5GB
- Checkpoints: ~700MB each (keeps last 3)
- **Total needed**: ~6GB during setup, ~3GB after cleanup

## Cleanup (Optional)

After preparing the dataset, you can remove the raw files:
```bash
rm -rf ./data/wikipedia/wikipedia-2k-high-quality/
```

The processed dataset in `./data/processed_dataset/` has everything needed for training.