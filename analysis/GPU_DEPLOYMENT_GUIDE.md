# GPU Deployment Guide for Yxanul 0.6B

## Quick Start (Copy-Paste Commands)

### Step 1: SSH into your GPU instance
```bash
ssh user@your-gpu-instance
```

### Step 2: Start NVIDIA NGC Container with GPU
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.10-py3
```

### Step 3: Inside the container, run setup
```bash
# Install git-lfs (required for dataset)
apt-get update && apt-get install -y git-lfs
git lfs install

# Clone Yxanul repository
git clone https://github.com/yourusername/yxanul_0.6B.git
cd yxanul_0.6B

# Install additional requirements
pip install transformers datasets accelerate tensorboard wandb

# Download FineWeb-Edu dataset (10.87 GiB)
git clone https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
cd fineweb-edu
git lfs pull --include="data/CC-MAIN-2025/*.parquet"
cd ..
mkdir -p fineweb-edu-highest-quality-2025/data
mv fineweb-edu/data/CC-MAIN-2025/*.parquet fineweb-edu-highest-quality-2025/data/
```

### Step 4: Start Training

## Training Entry Points

### **Option 1: MAXIMUM SPEED** (Recommended for RTX 4090)
```bash
python train_fp8.py --config configs/fineweb_training_ultra_curriculum.yaml
```
- Uses **FP8 mixed precision** (2x speedup)
- **Ultra-aggressive curriculum** (8→2048 tokens)
- **SuperBPE-t80k** tokenizer
- Expected: **60,000 tokens/sec** on RTX 4090
- Training time: **6-10 hours**

### Option 2: Standard BF16 with Curriculum
```bash
python train_curriculum.py --config configs/fineweb_training_ultra_curriculum.yaml
```
- Uses **BF16 precision** (no FP8)
- **Ultra-aggressive curriculum**
- Use this if Transformer Engine is not available
- Expected: **35,000 tokens/sec**

### Option 3: Basic Training (No Curriculum)
```bash
python train.py --config configs/fineweb_training.yaml
```
- Simple BF16 training
- Fixed batch size and sequence length
- For baseline comparisons

## Training Scripts Explained

| Script | Model | Trainer | Precision | Curriculum | Speed |
|--------|-------|---------|-----------|------------|-------|
| `train_fp8.py` | model_fp8_optimized.py | FP8Trainer | FP8/BF16 | Yes | Fastest |
| `train_curriculum.py` | model.py | EnhancedTrainer | BF16 | Yes | Fast |
| `train.py` | model.py | Trainer | BF16 | No | Standard |

## Key Differences

### train_fp8.py
- Imports: `from model_fp8_optimized import create_fp8_model`
- Trainer: `FP8Trainer` with Transformer Engine support
- Fallback: Automatically uses BF16 if TE not available
- Best for: Production training on modern GPUs

### train_curriculum.py  
- Imports: `from model import create_model`
- Trainer: `EnhancedTrainer` with curriculum stages
- No FP8 support (pure BF16)
- Best for: When TE is not available

## Monitoring Training

In another terminal:
```bash
# View logs
tensorboard --logdir logs/ --bind_all

# Watch GPU usage
nvidia-smi -l 1

# Check training progress
tail -f logs/training.log
```

## Resume Training

```bash
# Resume from latest checkpoint
python train_fp8.py \
  --config configs/fineweb_training_ultra_curriculum.yaml \
  --resume_from_checkpoint checkpoints/latest.pt
```

## Common Issues

### "Transformer Engine not found"
- Solution: Use `train_curriculum.py` instead of `train_fp8.py`
- The model will use BF16 instead of FP8

### "CUDA out of memory"
- Reduce batch sizes in config file
- RTX 4090 (24GB) should handle default settings

### "Dataset not found"
- Ensure parquet files are in `fineweb-edu-highest-quality-2025/data/`
- Should have 300 parquet files totaling 10.87 GiB

## GPU Memory Requirements

| Config | Batch Size | Seq Length | Memory Usage |
|--------|------------|------------|--------------|
| Stage 1 | 768 | 8 | ~8 GB |
| Stage 5 | 96 | 128 | ~12 GB |
| Stage 10 | 12 | 2048 | ~20 GB |

## Expected Results

- **Perplexity < 50**: Within 2 hours
- **Perplexity < 30**: Within 6 hours  
- **Final perplexity**: ~25-28 after 3 epochs
- **Training time**: 6-10 hours total on RTX 4090