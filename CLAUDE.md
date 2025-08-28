# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is Yxanul 270M/0.6B, a language model implementation repository. **IMPORTANT: All active development happens in the `experimental/` directory**, which contains clean, working implementations optimized for TinyStories dataset training and various architecture experiments.

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

## Experimental Directory (Active Development)

All current work is in `experimental/`:

### Core Files
- **model.py**: Clean GPT-2 architecture with GQA, RoPE, SwiGLU
  - Supports factorized embeddings (rank=128) reducing params from 113M → 87M with GPT-2 vocab
  - With SuperBPE 200k vocab: 381M → 125M params via factorization
- **train_tinystories.py**: Main training script
  - Supports both GPT-2 (50k) and SuperBPE (200k) vocabularies
  - Factorized embeddings via `--factorized --embedding_rank 128`
  - SuperBPE mode via `--superbpe` flag
- **prepare_tinystories_superbpe.py**: Tokenizes TinyStories with SuperBPE
  - Achieves 42.9% token reduction (301M → 172M tokens)
  - 1.75x effective speedup in training

### Key Experimental Findings
1. **SuperBPE Performance**: 200k vocabulary achieves 42.9% token reduction but is memory-bandwidth limited
2. **Memory Bottleneck**: RTX 4090 hits 91% memory bandwidth with 200k vocab (64k tok/s vs 110k with 50k vocab)
3. **Optimal Settings (RTX 4090, 24GB)**:
   - SuperBPE: batch_size=32, block_size=128, grad_accum=32
   - GPT-2: batch_size=64, block_size=128, grad_accum=16
4. **Factorized Embeddings Essential**: For 200k vocab, reduces embedding params by 6x (154M → 26M)

## Main Directory (Production - Currently Inactive)

The root directory contains the production 270M model with FP8 support via TransformerEngine:
- **FP8 Training**: `train_fp8.py` with `src/model_fp8_optimized.py`
- **Data Pipeline**: `src/data_pipeline.py` - SuperBPE variants for FineWeb dataset

## Common Development Commands

### Experimental Training (Active Development)

```bash
cd experimental/

# Train with GPT-2 tokenizer + factorized embeddings (87M params)
python train_tinystories.py --factorized --embedding_rank 128 --max_iters 1000

# Prepare SuperBPE dataset (42.9% token reduction) 
python prepare_tinystories_superbpe.py

# Train with SuperBPE tokenizer + factorized embeddings (125M params)
python train_tinystories.py --superbpe --factorized --embedding_rank 128 --max_iters 1000

# Test generation from checkpoint
python test_generation.py checkpoints_tinystories/best_model.pt
```

### Production Training (Currently Inactive)

```bash
# Main FP8 training script
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

