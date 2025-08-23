#!/bin/bash

# ============================================
# Yxanul 0.6B Stable Setup Script for Vast.ai/RunPod
# Works around Python 3.12 compatibility issues
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Yxanul 0.6B Training Setup (Stable Version)${NC}"
echo -e "${BLUE}For Vast.ai/RunPod Instances${NC}"
echo -e "${BLUE}============================================${NC}"

# Function to print colored status
print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Detect Python version
echo -e "\n${BLUE}[Step 1/8] Checking Python version...${NC}"

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "Python version: $PYTHON_VERSION"

# Check if Python 3.12 (problematic for Transformer Engine)
if [[ "$PYTHON_MINOR" -eq 12 ]]; then
    print_warning "Python 3.12 detected - Transformer Engine has compatibility issues"
    print_warning "Will use BF16 training instead of FP8 (still very fast!)"
    USE_FP8="false"
else
    print_status "Python $PYTHON_VERSION is compatible with Transformer Engine"
    USE_FP8="true"
fi

# 2. Detect GPU and CUDA
echo -e "\n${BLUE}[Step 2/8] Detecting GPU and CUDA...${NC}"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
echo "GPU: $GPU_NAME ($GPU_MEMORY MB)"

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
echo "CUDA Version: $CUDA_VERSION"

# Determine PyTorch CUDA version
if [[ "$CUDA_MAJOR" -eq 12 ]]; then
    if [[ "$CUDA_MINOR" -ge 4 ]]; then
        PYTORCH_CUDA="cu124"
        print_status "Using CUDA 12.4 packages"
    elif [[ "$CUDA_MINOR" -ge 1 ]]; then
        PYTORCH_CUDA="cu121"
        print_status "Using CUDA 12.1 packages"
    else
        PYTORCH_CUDA="cu121"
        print_warning "CUDA 12.0 detected, using cu121 packages"
    fi
elif [[ "$CUDA_MAJOR" -eq 11 ]] && [[ "$CUDA_MINOR" -eq 8 ]]; then
    PYTORCH_CUDA="cu118"
    print_status "Using CUDA 11.8 packages"
else
    print_error "Unsupported CUDA version: $CUDA_VERSION"
    print_warning "Attempting to use cu121 packages"
    PYTORCH_CUDA="cu121"
fi

# 3. Clone repository
echo -e "\n${BLUE}[Step 3/8] Setting up repository...${NC}"

if [ -d "yxanul_0.6B" ]; then
    cd yxanul_0.6B
    git pull
    print_status "Repository updated"
else
    git clone https://github.com/yxanul/yxanul_0.6B.git
    cd yxanul_0.6B
    print_status "Repository cloned"
fi

# 4. Install PyTorch
echo -e "\n${BLUE}[Step 4/8] Installing PyTorch...${NC}"

# Clean previous installations
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch based on CUDA version
if [[ "$PYTORCH_CUDA" == "cu124" ]]; then
    # PyTorch 2.5.1 with CUDA 12.4
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
elif [[ "$PYTORCH_CUDA" == "cu121" ]]; then
    # PyTorch 2.5.1 with CUDA 12.1
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$PYTORCH_CUDA" == "cu118" ]]; then
    # PyTorch 2.5.1 with CUDA 11.8
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio
fi

# Verify PyTorch
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

print_status "PyTorch installed successfully"

# 5. Handle Transformer Engine based on Python version
echo -e "\n${BLUE}[Step 5/8] Setting up training acceleration...${NC}"

if [[ "$USE_FP8" == "true" ]]; then
    echo "Attempting to install Transformer Engine for FP8 support..."
    
    # Try to install Transformer Engine
    pip install ninja packaging
    
    # Use pre-built wheel for better compatibility
    if [[ "$PYTORCH_CUDA" == "cu121" ]] || [[ "$PYTORCH_CUDA" == "cu124" ]]; then
        # Try the latest stable version
        pip install transformer-engine==1.11.0 || {
            print_warning "Transformer Engine 1.11.0 failed, trying older version"
            pip install transformer-engine==1.10.0 || {
                print_warning "Transformer Engine installation failed"
                USE_FP8="false"
            }
        }
    else
        print_warning "Transformer Engine requires CUDA 12.1+"
        USE_FP8="false"
    fi
    
    # Test if it works
    if [[ "$USE_FP8" == "true" ]]; then
        python3 -c "import transformer_engine.pytorch as te; print('FP8 support: Available')" 2>/dev/null || {
            print_warning "Transformer Engine import failed, disabling FP8"
            USE_FP8="false"
        }
    fi
else
    print_warning "Skipping Transformer Engine due to Python 3.12"
fi

if [[ "$USE_FP8" == "false" ]]; then
    print_status "Will use BF16 training (still 3-4x faster with curriculum!)"
fi

# 6. Install other dependencies
echo -e "\n${BLUE}[Step 6/8] Installing dependencies...${NC}"

# Core packages with specific versions for stability
pip install transformers==4.46.3
pip install datasets==3.2.0
pip install accelerate==1.2.1
pip install tokenizers==0.20.3

# Training utilities
pip install wandb
pip install tqdm
pip install pyyaml
pip install numpy
pip install pandas

# Optional optimizations
pip install bitsandbytes || print_warning "bitsandbytes failed (optional)"
pip install triton || print_warning "triton failed (optional)"

print_status "Dependencies installed"

# 7. Configure WandB
echo -e "\n${BLUE}[Step 7/8] Configuring WandB...${NC}"

wandb login --relogin 4444c18d3905dde9ab69774b2322a0c41ab209d3 || print_warning "WandB login failed"

# 8. Create training scripts
echo -e "\n${BLUE}[Step 8/8] Creating helper scripts...${NC}"

# Create directories
mkdir -p checkpoints logs data

# Create appropriate start script based on FP8 availability
if [[ "$USE_FP8" == "true" ]]; then
    cat > start_training.sh << 'EOF'
#!/bin/bash
echo "Starting FP8-accelerated training on FineWeb-Edu..."
python train_fp8.py --config configs/stage1_curriculum_fp8_fineweb.yaml
EOF
else
    cat > start_training.sh << 'EOF'
#!/bin/bash
echo "Starting BF16 training on FineWeb-Edu (FP8 not available)..."
# Update config to use FineWeb dataset
if ! grep -q "fineweb-edu-highest-quality" configs/stage1_curriculum_optimized_24gb.yaml; then
    sed -i 's|dataset_name:.*|dataset_name: "Yxanul/fineweb-edu-highest-quality-2025"|' configs/stage1_curriculum_optimized_24gb.yaml
fi
python train_curriculum.py --config configs/stage1_curriculum_optimized_24gb.yaml
EOF
fi

chmod +x start_training.sh

# Create monitor script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
watch -n 1 'nvidia-smi; echo ""; tail -n 20 logs/training.log 2>/dev/null || echo "Waiting for training to start..."'
EOF
chmod +x monitor_training.sh

# Final verification
echo -e "\n${BLUE}Running verification...${NC}"

cat > verify.py << 'EOF'
import torch
print("="*50)
print("VERIFICATION RESULTS")
print("="*50)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

try:
    import transformer_engine.pytorch as te
    print("FP8: Available (2x speedup)")
except:
    print("FP8: Not available (using BF16 instead)")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except:
    print("Transformers: ERROR")

print("="*50)
EOF

python3 verify.py

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
if [[ "$USE_FP8" == "true" ]]; then
    echo "FP8 support: ENABLED (2x speedup over BF16)"
else
    echo "FP8 support: DISABLED (using BF16 - still fast!)"
    echo "Note: This is due to Python 3.12 compatibility issues"
fi
echo ""
echo "Commands:"
echo "  ./start_training.sh   - Start training"
echo "  ./monitor_training.sh - Monitor GPU usage"
echo ""
echo "Dataset: Yxanul/fineweb-edu-highest-quality-2025"
echo "Size: 4.1B tokens (will download on first run)"
echo ""
print_status "Ready to train!"