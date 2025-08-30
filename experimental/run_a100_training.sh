#!/bin/bash
# Optimal training configuration for A100 80GB with SmolLM tokenizer

echo "=========================================="
echo "A100 80GB Training Configuration"
echo "SmolLM Tokenizer (49,152 vocab)"
echo "=========================================="

# First, prepare the dataset if not already done
if [ ! -f "data_fineweb_edu_smollm/train.bin" ]; then
    echo "Preparing SmolLM dataset (this will take ~30 minutes)..."
    python prepare_fineweb_edu_highest.py
fi

# Run training with optimal A100 settings
python train_tinystories.py \
    --data_dir data_fineweb_edu_smollm \
    --vocab_size 49152 \
    --block_size 2048 \
    --batch_size 64 \
    --gradient_accumulation_steps 4 \
    --max_iters 10000 \
    --wandb_run_name "smollm-a100-2048ctx"

# Expected performance:
# - Memory usage: ~22GB / 80GB (lots of headroom)
# - Speed: ~200-250k tokens/sec
# - Effective batch size: 64 * 4 = 256
# - Tokens per iteration: 256 * 2048 = 524,288
# - Total tokens: 10000 * 524,288 = 5.2B tokens
# - Training time: ~6-8 hours
# - Cost at $0.628/hr: ~$4-5 total