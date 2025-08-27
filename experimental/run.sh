#!/bin/bash
# Simple run script - prepare data then train

echo "========================================="
echo "Simple, Clean Training"
echo "========================================="

# Check if data exists
if [ ! -f "experimental/data/train.bin" ]; then
    echo "Preparing data..."
    python experimental/prepare_data.py
else
    echo "Data already prepared, skipping..."
fi

echo ""
echo "Starting training..."
echo ""

# Set environment for better performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Run training
python experimental/train.py

echo ""
echo "Training complete!"