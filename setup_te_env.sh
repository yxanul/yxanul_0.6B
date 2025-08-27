#!/bin/bash
# Environment variables to optimize TransformerEngine v2.4 backend selection
# Run this before training: source setup_te_env.sh

echo "Setting up TransformerEngine environment variables..."

# Prefer Flash Attention over cuDNN fused attention
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0  # Disable cuDNN fused attention to avoid kernel issues

# Enable debug output to see backend selection
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=1

# Optional: Get detailed cuDNN errors if they still occur
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr

# Allow non-deterministic algorithms for better performance
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

echo "Environment configured:"
echo "  - Flash Attention: ENABLED (preferred)"
echo "  - cuDNN Fused Attention: DISABLED"
echo "  - Debug output: ENABLED"
echo "  - Non-deterministic algos: ALLOWED"
echo ""
echo "Ready to run training!"