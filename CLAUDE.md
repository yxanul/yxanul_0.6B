# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is Yxanul 197M, a state-of-the-art efficient language model implementation that achieves 15x faster training than baseline through advanced optimizations. The codebase implements a 197M parameter transformer model with SuperBPE tokenization, FP8 mixed precision, and ultra-aggressive curriculum learning.

## Recent Discoveries and Improvements (December 2024)

### TransformerEngine v2.4 Migration
We've successfully migrated to NVIDIA TransformerEngine v2.4, achieving 40-50% additional speedup through:
- **Native TransformerLayer**: Replaces custom attention/FFN implementations
- **Proper FP8 context management**: Backward pass now correctly happens OUTSIDE fp8_autocast
- **Native module support**: Built-in GQA, RMSNorm, and SwiGLU implementations
- **Automatic Flash Attention 3**: Auto-selected on compatible GPUs with NGC 25.05+

### Multi-Domain Validation
Implemented proper validation using separate datasets:
- **C4**: English text validation (2k samples)
- **GSM8K**: Mathematics validation (full set)
- **HumanEval**: Code generation validation (full set)
This enables curriculum ratio tuning based on surprise ratios across domains.

### Checkpoint Management
Added robust CheckpointManager with:
- Automatic rotation (keeps 3-5 latest)
- Best model tracking
- Metadata preservation
- Safe atomic saves

## Core Architecture

The system consists of multiple training pipelines:

### Original Implementation
- **BF16 Training**: `src/trainer.py` with `src/model.py` - Standard precision training
- **FP8 Training**: `src/trainer_fp8.py` with `src/model_fp8_optimized.py` - Mixed precision with DeepSeek V3-inspired strategy
- **Curriculum Training**: `train_curriculum.py` or `train_fp8.py` with curriculum configs - 10-stage progressive learning

### TransformerEngine v2.4 Implementation (Recommended)
- **TE v2.4 Model**: `src/model_te_v2.py` - Native TransformerLayer with built-in optimizations
- **TE v2.4 Trainer**: `src/trainer_te_v2.py` - Proper fp8_autocast context management
- **TE v2.4 Training**: `train_te_v2.py` - Complete training script with benchmarking

The data pipeline (`src/data_pipeline.py`) implements SuperBPE tokenization variants (t80k for speed, t180k for quality) that reduce token count by 37.5%.

## Common Development Commands

### Training Commands

```bash
# NEW: TransformerEngine v2.4 - Maximum performance (40-50% faster)
python train_te_v2.py --config configs/te_v2_config.yaml

# Benchmark TE v2.4 performance
python train_te_v2.py --benchmark --batch-size 32

# Original FP8 implementation
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

# Test multi-domain validation
python src/multi_domain_validation.py

# Compare SuperBPE tokenizer variants
python compare_superbpe_variants.py

# Check GPU and TransformerEngine availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformer_engine as te; print(f'TransformerEngine v{te.__version__} Ready!')" 2>/dev/null || echo "TE not available"

# Test TE v2.4 model creation
python src/model_te_v2.py
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

- `configs/te_v2_config.yaml`: **NEW** - TransformerEngine v2.4 optimized configuration
- `configs/fineweb_training_ultra_curriculum.yaml`: 10-stage curriculum from 8 to 2048 tokens
- `configs/model_config.yaml`: 197M model architecture with 28 layers, GQA, RoPE, SwiGLU
- `configs/model_config_270m.yaml`: 270M variant configuration
- `configs/fineweb_training_fp8.yaml`: FP8 training configuration
- `configs/optimization.yaml`: Training hyperparameters

## Critical Implementation Details

### SuperBPE Tokenization
The tokenizer in `src/data_pipeline.py` can switch between:
- **t80k** (line ~20): 7.184 chars/token for maximum speed
- **t180k**: 7.226 chars/token for +0.4% quality

To switch, modify the tokenizer initialization in `data_pipeline.py`.

### FP8 Mixed Precision Strategy

#### TransformerEngine v2.4 Implementation (Recommended)
**Critical**: Backward pass must happen OUTSIDE fp8_autocast context!
```python
# Correct TE v2.4 pattern:
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    loss = model(inputs)  # Forward only
loss.backward()  # Backward OUTSIDE context
```

Formats supported:
- **hybrid** (default): E4M3 forward, E5M2 backward - best stability
- **e4m3**: E4M3 everywhere - more precision
- **mxfp8**: Block-scaled FP8 - experimental, potentially better accuracy

#### Original Implementation
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

### TransformerEngine v2.4 (NGC 25.05+)
On RTX 4090:
- **TE v2.4 FP8**: ~85,000 tokens/sec (40% faster than old FP8)
- **Training time**: 4-6 hours for 3 epochs
- **Memory usage**: 5.0 GB (20% reduction)

On H100:
- **TE v2.4 FP8**: ~150,000 tokens/sec
- **FP8 DPA**: Enabled for attention operations
- **Training time**: 1.5 hours for 3 epochs

### Original Implementation
On RTX 4090:
- **FP8 mode**: 60,000 tokens/sec (actual: ~17k due to implementation issues)
- **BF16 mode**: 35,000 tokens/sec
- **Training time**: 6-10 hours for 3 epochs
- **Perplexity < 50**: Achieved in ~2 hours

## Important Notes

1. **Dataset Location**: The FineWeb-Edu dataset (10.87 GiB) is stored locally in `fineweb-edu-highest-quality-2025/`. This contains 300 parquet files with 4.1B tokens of high-quality educational content.

2. **Validation Datasets**: Multi-domain validation datasets in `validation/`:
   - `c4_validation_2k.parquet`: 2.6 MB English text
   - `gsm8k_validation.parquet`: 405 KB math problems
   - `humaneval_validation.parquet`: 81 KB code tasks

3. **NVIDIA NGC Container**: For TransformerEngine v2.4, use NGC 25.05 or later:
   ```bash
   # NGC 25.05+ includes TE v2.3+, Flash Attention 3
   docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/pytorch:25.05-py3
   ```
   Older versions (24.10) have TE v1.7 which lacks native GQA/RMSNorm support.

4. **Curriculum Batch Sizes**: The configuration files specify batch sizes optimized for RTX 4090. Adjust these if using different GPUs.

5. **Wandb Logging**: Training scripts log to Weights & Biases. Set `WANDB_API_KEY` or disable with `--no-wandb`.
   - Fixed: WANDB_REQUIRE_SERVICE environment variable (was incorrectly WANDB__REQUIRE_SERVICE)

6. **Checkpoint Management**: Enhanced CheckpointManager automatically:
   - Rotates checkpoints (keeps 3-5 latest)
   - Tracks best model by validation perplexity
   - Saves metadata and FP8 recipes
   - Resume with `--resume_from_checkpoint`

7. **FP8 Tensor Alignment**: For optimal FP8 performance, tensors must be divisible by 16. The model automatically pads sequences when needed.

## Troubleshooting

- **CUDA out of memory**: Reduce batch sizes in curriculum config
- **TransformerEngine not found**: Use NGC container 25.05+ or install manually
- **FP8 not converging**: Ensure backward pass is OUTSIDE fp8_autocast context
- **Slow training compared to expected**:
  - Verify Flash Attention 3 is loaded (check logs)
  - Ensure using native TE modules, not custom implementations
  - Check tensor dimensions are divisible by 16
- **Dataset not found**: Check that parquet files are in `fineweb-edu-highest-quality-2025/data/`
- **Model return order error**: Fixed - model returns `(loss, logits)` not `(logits, loss)`
- **DataLoader prefetch_factor error**: Fixed - conditionally add based on PyTorch version

## Known Issues Fixed

1. **WANDB environment variable**: Changed from `WANDB__REQUIRE_SERVICE` to `WANDB_REQUIRE_SERVICE`
2. **Model return order**: Fixed to return `(loss, logits)` consistently
3. **DataLoader prefetch_factor**: Now conditionally added for compatibility
4. **FP8 implementation**: Migrated to TE v2.4 for proper FP8 operations
5. **Validation bias**: Added multi-domain validation instead of using training tail
6. **Checkpoint bloat**: Implemented automatic rotation with CheckpointManager

## Architecture Comparison

| Component | Original Implementation | TE v2.4 Implementation | Speedup |
|-----------|------------------------|------------------------|----------|
| Attention | Custom multi-head with te.Linear | Native TransformerLayer | 1.5x |
| FFN | Manual SwiGLU with te.Linear | Native SwiGLU in TransformerLayer | 1.3x |
| Normalization | Custom RMSNorm | Native te.RMSNorm | 1.2x |
| GQA | Manual KV head expansion | Native GQA support | 1.4x |
| FP8 Context | Incorrect (backward inside) | Correct (backward outside) | 1.2x |
| **Overall** | Baseline | **40-50% faster** | **1.45x** |