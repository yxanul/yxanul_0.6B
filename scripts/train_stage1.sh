#!/bin/bash
# Launch script for Stage 1: Wikipedia training on 8x A100/H100

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Performance optimizations
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Triton optimizations for torch.compile
export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

echo "=========================================="
echo "Yxanul 0.6B Training - Stage 1: Wikipedia"
echo "=========================================="
echo "GPUs: 8x A100/H100"
echo "Model: 28 layers, 768 hidden, ~500M params"
echo "Dataset: Wikipedia 2K+ filtered"
echo "=========================================="

# Launch with torchrun (PyTorch native)
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    --master_addr=localhost \
    ../src/trainer.py \
    --config_dir ../configs \
    --stage stage1_wikipedia \
    --num_epochs 15

# Alternative: Launch with DeepSpeed
# deepspeed --num_gpus=8 \
#     --master_port=29500 \
#     ../src/trainer.py \
#     --config_dir ../configs \
#     --stage stage1_wikipedia \
#     --deepspeed ../configs/deepspeed_config.json

echo "Training completed!"