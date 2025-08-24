# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is Yxanul 197M, a state-of-the-art efficient language model implementation that achieves 15x faster training than baseline through advanced optimizations. The codebase implements a 197M parameter transformer model with SuperBPE tokenization, FP8 mixed precision, and ultra-aggressive curriculum learning.

## Core Architecture

The system consists of three main training pipelines:
- **BF16 Training**: `src/trainer.py` with `src/model.py` - Standard precision training
- **FP8 Training**: `src/trainer_fp8.py` with `src/model_fp8_optimized.py` - Mixed precision with DeepSeek V3-inspired strategy
- **Curriculum Training**: `train_curriculum.py` or `train_fp8.py` with curriculum configs - 10-stage progressive learning

The data pipeline (`src/data_pipeline.py`) implements SuperBPE tokenization variants (t80k for speed, t180k for quality) that reduce token count by 37.5%.

## Common Development Commands

### Training Commands

```bash
# Research mode - Maximum speed with FP8 + Ultra Curriculum + SuperBPE-t80k
python train_fp8.py --config configs/fineweb_training_ultra_curriculum.yaml

# Standard BF16 training without curriculum
python train.py --config configs/fineweb_training.yaml

# Curriculum learning with BF16
python train_curriculum.py --config configs/fineweb_training_ultra_curriculum.yaml

# Monitor training progress
tensorboard --logdir logs/
```

### Testing and Validation

```bash
# Test dataset loading
python test_dataset_loading.py

# Compare SuperBPE tokenizer variants
python compare_superbpe_variants.py

# Check GPU and FP8 availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformer_engine; print('FP8 Ready!')" 2>/dev/null || echo "FP8 not available"
```

### Environment Setup

```bash
# Install requirements (if not using NGC container)
pip install -r requirements.txt

# Verify dataset is present
ls fineweb-edu-highest-quality-2025/data/*.parquet | wc -l
# Should show: 300
```

## Key Configuration Files

- `configs/fineweb_training_ultra_curriculum.yaml`: 10-stage curriculum from 8 to 2048 tokens
- `configs/model_config.yaml`: 197M model architecture with 28 layers, GQA, RoPE, SwiGLU
- `configs/fineweb_training_fp8.yaml`: FP8 training configuration
- `configs/optimization.yaml`: Training hyperparameters

## Critical Implementation Details

### SuperBPE Tokenization
The tokenizer in `src/data_pipeline.py` can switch between:
- **t80k** (line ~20): 7.184 chars/token for maximum speed
- **t180k**: 7.226 chars/token for +0.4% quality

To switch, modify the tokenizer initialization in `data_pipeline.py`.

### FP8 Mixed Precision Strategy
When using FP8 training:
- 60% operations in FP8 (E4M3): Matrix multiplies, FFN, attention projections
- 40% operations in BF16: Embeddings, normalizations, output head
- <1% in FP32: Optimizer states, loss computation

### Curriculum Learning Stages
The ultra-curriculum starts from just 8 tokens (meaningful with SuperBPE):
1. Stage 1-2 (8-16 tokens): Basic patterns, phrases
2. Stage 3-4 (32-64 tokens): Sentence structure
3. Stage 5-6 (128-256 tokens): Paragraph coherence
4. Stage 7-8 (512-768 tokens): Document structure
5. Stage 9-10 (1536-2048 tokens): Long-range dependencies

### Memory Requirements
- RTX 4090 (24GB): Can handle all configurations
- Batch sizes automatically adjusted per curriculum stage
- Model + Optimizer ≈ 2.1GB in FP8 mode

## Performance Expectations

On RTX 4090:
- **FP8 mode**: 60,000 tokens/sec
- **BF16 mode**: 35,000 tokens/sec
- **Training time**: 6-10 hours for 3 epochs
- **Perplexity < 50**: Achieved in ~2 hours

## Important Notes

1. **Dataset Location**: The FineWeb-Edu dataset (10.87 GiB) is stored locally in `fineweb-edu-highest-quality-2025/`. This contains 300 parquet files with 4.1B tokens of high-quality educational content.

2. **NVIDIA NGC Container**: For best results, use the NVIDIA container which includes PyTorch, CUDA, Transformer Engine, and Flash Attention pre-installed:
   ```bash
   docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/pytorch:24.10-py3
   ```

3. **Curriculum Batch Sizes**: The configuration files specify batch sizes optimized for RTX 4090. Adjust these if using different GPUs.

4. **Wandb Logging**: Training scripts log to Weights & Biases. Set `WANDB_API_KEY` or disable with `--no-wandb`.

5. **Checkpoint Management**: Models are saved to `checkpoints/` every 10,000 steps. Resume training with `--resume_from_checkpoint`.

## Troubleshooting

- **CUDA out of memory**: Reduce batch sizes in curriculum config
- **FP8 not available**: Fall back to BF16 training (still fast)
- **Slow training**: Ensure Flash Attention is enabled and using correct tokenizer variant
- **Dataset not found**: Check that parquet files are in `fineweb-edu-highest-quality-2025/data/`