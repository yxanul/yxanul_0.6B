# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is Yxanul 270M, a language model implementation focused on FP8 mixed precision training. The codebase implements a 270M parameter transformer model with SuperBPE tokenization (200k vocabulary), FP8 mixed precision via TransformerEngine, and curriculum learning.

## Critical Bug Fix (December 2024)

### Next-Token Prediction Fixed
Fixed critical bug where model was predicting current token instead of next token:
- **Bug**: Model computed `CE(logits, labels)` without shifting - essentially a copy task
- **Fix**: Now properly computes `CE(logits[:-1], labels[1:])` for true next-token prediction
- **Impact**: Loss values now realistic (starting ~12, converging to ~9) instead of artificially low

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

### Current Implementation (Use These)
- **FP8 Training**: `train_fp8.py` with `src/model_fp8_optimized.py` - Mixed precision with DeepSeek V3-inspired strategy
- **FP8 Trainer**: `src/trainer_fp8.py` - Extends EnhancedTrainer with FP8 support
- **Model**: `src/model_fp8_optimized.py` - 270M model with GQA, RoPE, SwiGLU
- **Curriculum Training**: Configured via `configs/fineweb_training_fp8.yaml`

### Deprecated Files (Do Not Use)
- `src/model_te_v2.py` - TE v2.4 implementation (abandoned)
- `src/trainer_te_v2.py` - TE v2.4 trainer (abandoned)
- `train_te_v2.py` - TE v2.4 training script (abandoned)

The data pipeline (`src/data_pipeline.py`) implements SuperBPE tokenization variants (t80k for speed, t180k for quality) that reduce token count by 37.5%.

## Common Development Commands

### Training Commands

```bash
# Main FP8 training script (RECOMMENDED)
python train_fp8.py --config configs/fineweb_training_fp8.yaml

# Train without FP8 (BF16 only)
python train_fp8.py --config configs/fineweb_training_fp8.yaml --disable-fp8

# Evaluation only
python train_fp8.py --config configs/fineweb_training_fp8.yaml --eval-only

# Resume from checkpoint
python train_fp8.py --config configs/fineweb_training_fp8.yaml --checkpoint checkpoints/latest.pt

# Monitor training progress
tensorboard --logdir logs/
```

### Testing and Validation

```bash
# Test dataset loading
python test_dataset_loading.py

# Test multi-domain validation
python src/multi_domain_validation.py

# Check GPU and TransformerEngine availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformer_engine as te; print(f'TransformerEngine Ready!')" 2>/dev/null || echo "TE not available"
```

### Environment Setup

```bash
# Install requirements
pip install -r requirements.txt

# Verify dataset is present
ls experimental-pretrain-1b/*.parquet
# Should show: dataset_1b.parquet
```

## Key Configuration Files

- `configs/fineweb_training_fp8.yaml`: Main FP8 training configuration with curriculum
- `configs/optimization.yaml`: Training hyperparameters (torch.compile disabled for FP8)
- `configs/model_config.yaml`: Model architecture (ignored - hardcoded 270M in train_fp8.py)
- `configs/model_config_270m.yaml`: 270M variant (reference only)

## Critical Implementation Details

### SuperBPE Tokenization
The tokenizer in `src/data_pipeline.py` can switch between:
- **t80k** (line ~20): 7.184 chars/token for maximum speed
- **t180k**: 7.226 chars/token for +0.4% quality

To switch, modify the tokenizer initialization in `data_pipeline.py`.

### FP8 Mixed Precision Strategy

#### Current Implementation
- 60% operations in FP8 (E4M3): Matrix multiplies, FFN, attention projections
- 40% operations in BF16: Embeddings, normalizations, output head
- <1% in FP32: Optimizer states, loss computation

### Curriculum Learning Stages (configs/fineweb_training_fp8.yaml)
Memory-optimized for RTX 5090 (32GB VRAM):
1. Stage 1 (0-3k steps): 128 tokens, batch_size=8
2. Stage 2 (3k-6k): 256 tokens, batch_size=4
3. Stage 3 (6k-10k): 512 tokens, batch_size=2
4. Stage 4 (10k-15k): 768 tokens, batch_size=2
5. Stage 5 (15k-25k): 1024 tokens, batch_size=1
6. Stage 6 (25k+): 2048 tokens, batch_size=1

### Memory Requirements
- RTX 5090 (32GB): ~28GB used with 200k vocabulary
- Must use batch_size=1 for training, validation
- Gradient accumulation=32 for effective batch size

## Performance Expectations

### Current Implementation (RTX 5090)
- **BF16 mode**: ~3,000 tokens/sec (effective with grad_accum=32)
- **FP8 mode**: Target 2x speedup over BF16 (when working)
- **Loss progression**: 12.3→9.7 in 1400 steps (healthy convergence)
- **Memory usage**: ~28GB with 200k vocabulary

## Important Notes

1. **Dataset Location**: The experimental-pretrain-1b dataset is stored locally in `experimental-pretrain-1b/dataset_1b.parquet`. This contains 637,270 examples with 1B tokens total.

2. **Validation Datasets**: Multi-domain validation datasets in `validation/`:
   - `c4_validation_2k.parquet`: 2.6 MB English text
   - `gsm8k_validation.parquet`: 405 KB math problems
   - `humaneval_validation.parquet`: 81 KB code tasks

3. **TransformerEngine**: Install via pip or use NGC container. Required for FP8 training.

4. **Curriculum Batch Sizes**: The configuration files specify batch sizes optimized for RTX 5090 (32GB). Already reduced to prevent OOM.

5. **Wandb Logging**: Training scripts log to Weights & Biases. Set `WANDB_API_KEY` or disable with `--no-wandb`.
   - Fixed: WANDB_REQUIRE_SERVICE environment variable (was incorrectly WANDB__REQUIRE_SERVICE)

6. **Checkpoint Management**: Enhanced CheckpointManager automatically:
   - Rotates checkpoints (keeps 3-5 latest)
   - Tracks best model by validation perplexity
   - Saves metadata and FP8 recipes
   - Resume with `--resume_from_checkpoint`

7. **FP8 Tensor Alignment**: For optimal FP8 performance, tensors must be divisible by 16. The model automatically pads sequences when needed.

## Troubleshooting

- **CUDA out of memory**: Already fixed - batch sizes reduced, validation uses batch_size=1
- **TransformerEngine not found**: Install via pip or use NGC container
- **Loss too low (< 2.0)**: Fixed - model now properly predicts next token, not current
- **Slow tokens/sec display**: Shows ~90 instead of ~3k - display issue only, training is correct
- **Dataset not found**: Check that `experimental-pretrain-1b/dataset_1b.parquet` exists
- **Model return order error**: Fixed - model returns `(loss, logits)` not `(logits, loss)`
- **DataLoader prefetch_factor error**: Fixed - conditionally add based on PyTorch version

## Known Issues Fixed

1. **Next-token prediction bug**: Fixed - model was predicting current token (copy task)
2. **OOM on validation**: Fixed - reduced batch_size to 1, keep logits in BF16 for large vocab
3. **torch.compile conflicts**: Disabled in optimization.yaml for FP8 compatibility
4. **WandB hook conflicts**: Added WANDB_WATCH_DISABLED environment variable
5. **Model size**: Fixed - now correctly creates 270M model, not 350M
6. **Checkpoint management**: Automatic rotation with CheckpointManager
7. **Checkpoint saving error**: Fixed DelayedScaling recipe serialization (no 'format' attribute)
8. **Validation OOM with 200k vocab**: Limited batch_size to 2 for large vocabularies
9. **Double split argument**: Fixed curriculum dataloader passing split twice
10. **Attention mask ignored**: Changed from 'causal' to 'padding_causal' mask type
11. **Training instability**: Fixed factorized embedding init (0.02 vs 0.125) and added warmup
12. **FP8 calibration**: Increased from 10 to 64 steps for stability

