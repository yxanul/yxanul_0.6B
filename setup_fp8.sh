#!/bin/bash
# Setup script for FP8 training with Transformer Engine on RTX 4090

echo "============================================"
echo "Setting up FP8 Training with Transformer Engine"
echo "============================================"

# Check CUDA version
echo -e "\n[1/5] Checking CUDA version..."
nvcc --version

# Check PyTorch and CUDA availability
echo -e "\n[2/5] Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install Transformer Engine
echo -e "\n[3/5] Installing Transformer Engine..."
pip install transformer-engine[pytorch]

# Install additional dependencies
echo -e "\n[4/5] Installing additional dependencies..."
pip install apex-amp  # Optional but recommended for mixed precision
pip install ninja  # For faster compilation

# Verify installation
echo -e "\n[5/5] Verifying Transformer Engine installation..."
python -c "
import transformer_engine.pytorch as te
print('Transformer Engine version:', te.__version__)
print('FP8 available:', te.fp8.is_fp8_available())
print('FP8 supported:', te.fp8.get_fp8_support())
"

echo -e "\n============================================"
echo "Setup complete! FP8 training is ready."
echo "============================================"