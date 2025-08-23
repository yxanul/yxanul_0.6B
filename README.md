# Yxanul 177M Training

A clean, optimized training setup for the Yxanul 177M parameter language model using curriculum learning for 3-4x faster convergence.

## Quick Start (NVIDIA NGC Container)

### 1. Use NVIDIA Docker Image
```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.08-py3
```

This container includes:
- PyTorch with CUDA support
- Transformer Engine (FP8 support)
- Flash Attention 3
- All required dependencies

### 2. Clone and Start Training
```bash
# Clone repository
git clone https://github.com/yxanul/yxanul_0.6B.git
cd yxanul_0.6B

# Install additional packages
pip install transformers datasets wandb accelerate

# Configure WandB (optional)
wandb login

# Start training with FP8 (if supported)
python train_fp8.py --config configs/fineweb_training.yaml

# Or use BF16 training
python train_curriculum.py --config configs/fineweb_training.yaml
```

## Dataset

Training uses the FineWeb-Edu dataset:
- **Dataset**: `Yxanul/fineweb-edu-highest-quality-2025`
- **Size**: 4.1B tokens
- **Quality**: Educational score ≥3.5, min 1000 tokens per document

## Training Features

### Curriculum Learning (3-4x Speedup)
- Progressive sequence lengths: 128 → 2048 tokens
- Dynamic batch sizes to maintain constant GPU memory usage
- Adaptive learning rates per stage

### Performance
- **BF16**: ~20-25k tokens/second on RTX 4090
- **FP8**: ~35-40k tokens/second (with Transformer Engine)
- **Time to PPL<100**: ~10-12 hours

## Configuration

Main config: `configs/fineweb_training.yaml`

Key parameters:
- 8 curriculum stages for smooth progression
- Cosine learning rate schedule
- Gradient clipping per stage
- Automatic mixed precision (BF16/FP8)

## Project Structure

```
yxanul_0.6B/
├── configs/
│   ├── fineweb_training.yaml      # Main training config
│   ├── model_config.yaml          # Model architecture
│   └── optimization.yaml          # Optimizer settings
├── src/
│   ├── model.py                   # Standard model
│   ├── model_fp8.py              # FP8-optimized model
│   ├── trainer.py                # Base trainer
│   ├── trainer_fp8.py           # FP8 trainer
│   └── data_pipeline.py         # Data loading
├── train_curriculum.py           # BF16 training script
└── train_fp8.py                 # FP8 training script
```

## Requirements

- GPU: RTX 4090 or better (24GB VRAM)
- CUDA: 12.1+
- Python: 3.10 or 3.11 (3.12 has compatibility issues)
- Docker: NVIDIA Container Toolkit

## Notes

- The dataset will be downloaded automatically on first run
- Checkpoints are saved every 10,000 steps
- WandB logging is optional but recommended
- For issues, check the NGC container version compatibility

## License

MIT