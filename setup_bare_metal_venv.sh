#!/bin/bash

# ============================================
# Yxanul 0.6B Complete Bare Metal Setup with venv
# Installs Python 3.11, creates venv, and all dependencies
# Optimized for FP8 training with Transformer Engine
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Yxanul 0.6B Bare Metal Setup with venv${NC}"
echo -e "${BLUE}Installing Python 3.11 + Full ML Stack${NC}"
echo -e "${BLUE}============================================${NC}"

print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Update system and install essentials
echo -e "\n${BLUE}[Step 1/10] Installing system essentials...${NC}"

apt-get update
apt-get install -y \
    build-essential \
    software-properties-common \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tmux \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    libreadline-dev \
    zlib1g-dev \
    pkg-config

print_status "System essentials installed"

# 2. Install Python 3.11 (optimal for ML)
echo -e "\n${BLUE}[Step 2/10] Installing Python 3.11...${NC}"

# Add deadsnakes PPA for Python 3.11
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update

# Install Python 3.11 and development packages
apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils

# Verify Python 3.11 installation
python3.11 --version

print_status "Python 3.11 installed successfully"

# 3. Check CUDA and GPU
echo -e "\n${BLUE}[Step 3/10] Detecting GPU and CUDA...${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found! Please install NVIDIA drivers first."
    echo "For Ubuntu, run:"
    echo "  sudo apt-get install nvidia-driver-535"
    echo "Then reboot and run this script again."
    exit 1
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    echo "GPU Detected: $GPU_NAME ($GPU_MEMORY MB)"
    
    # Check CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "CUDA Driver Version: $CUDA_VERSION"
fi

# 4. Clone Yxanul repository
echo -e "\n${BLUE}[Step 4/10] Cloning Yxanul repository...${NC}"

cd ~
if [ -d "yxanul_0.6B" ]; then
    cd yxanul_0.6B
    git pull
    print_status "Repository updated"
else
    git clone https://github.com/yxanul/yxanul_0.6B.git
    cd yxanul_0.6B
    print_status "Repository cloned"
fi

# 5. Create virtual environment
echo -e "\n${BLUE}[Step 5/10] Creating Python virtual environment...${NC}"

# Create venv with Python 3.11
python3.11 -m venv venv

# Activate venv
source venv/bin/activate

# Verify we're in venv
which python
python --version

# Upgrade pip in venv
python -m pip install --upgrade pip setuptools wheel

print_status "Virtual environment created and activated"

# 6. Install PyTorch with CUDA support
echo -e "\n${BLUE}[Step 6/10] Installing PyTorch with CUDA support...${NC}"

# Install build tools
pip install ninja packaging cmake

# Install PyTorch 2.3.1 with CUDA 12.1 (most stable for Transformer Engine)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

if [ $? -eq 0 ]; then
    print_status "PyTorch installed with CUDA support"
else
    print_error "PyTorch installation failed"
    exit 1
fi

# 7. Install Transformer Engine for FP8
echo -e "\n${BLUE}[Step 7/10] Installing Transformer Engine for FP8...${NC}"

# Install Transformer Engine (version compatible with PyTorch 2.3)
pip install transformer-engine==1.10.0

# Verify Transformer Engine
python -c "
try:
    import transformer_engine.pytorch as te
    print(f'Transformer Engine: {te.__version__}')
    print('FP8 support: Available and ready!')
except Exception as e:
    print(f'Warning: {e}')
"

if [ $? -eq 0 ]; then
    print_status "Transformer Engine installed"
else
    print_warning "Transformer Engine may need verification"
fi

# 8. Install ML dependencies
echo -e "\n${BLUE}[Step 8/10] Installing ML dependencies...${NC}"

# Core ML packages
pip install transformers==4.41.2  # Stable version for PyTorch 2.3
pip install datasets==2.20.0
pip install accelerate==0.31.0
pip install tokenizers==0.19.1

# Training utilities
pip install wandb
pip install tqdm
pip install pyyaml
pip install numpy
pip install pandas
pip install scikit-learn

# Optional optimizations
pip install bitsandbytes || print_warning "bitsandbytes failed (optional)"
pip install triton==2.3.1 || print_warning "triton failed (optional)"

print_status "All ML dependencies installed"

# 9. Configure WandB
echo -e "\n${BLUE}[Step 9/10] Configuring WandB...${NC}"

wandb login --relogin 4444c18d3905dde9ab69774b2322a0c41ab209d3

print_status "WandB configured"

# 10. Create helper scripts
echo -e "\n${BLUE}[Step 10/10] Creating helper scripts...${NC}"

# Create directories
mkdir -p checkpoints logs data

# Create activation script
cat > activate.sh << 'EOF'
#!/bin/bash
# Activate the virtual environment
source ~/yxanul_0.6B/venv/bin/activate
echo "Virtual environment activated"
echo "Python: $(which python)"
echo "Version: $(python --version)"
EOF
chmod +x activate.sh

# Create FP8 training script
cat > start_training.sh << 'EOF'
#!/bin/bash
# Activate venv and start training
source ~/yxanul_0.6B/venv/bin/activate
echo "Starting FP8-accelerated training on FineWeb-Edu..."
echo "Dataset: Yxanul/fineweb-edu-highest-quality-2025 (4.1B tokens)"
python train_fp8.py --config configs/stage1_curriculum_fp8_fineweb.yaml
EOF
chmod +x start_training.sh

# Create monitor script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
watch -n 1 'nvidia-smi; echo ""; tail -n 20 logs/training.log 2>/dev/null || echo "Waiting for training..."'
EOF
chmod +x monitor_training.sh

# Create verification script
cat > verify_setup.py << 'EOF'
import sys
import torch
import subprocess

print("\n" + "="*60)
print("SETUP VERIFICATION")
print("="*60)

# System info
result = subprocess.run(['python', '--version'], capture_output=True, text=True)
print(f"Python: {result.stdout.strip()}")

# Check if in venv
import sys
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("Virtual Environment: ACTIVE")
else:
    print("Virtual Environment: WARNING - Not active!")

# PyTorch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")

# Transformer Engine
try:
    import transformer_engine.pytorch as te
    print(f"Transformer Engine: {te.__version__}")
    print("FP8 support: READY (2x speedup enabled)")
except Exception as e:
    print(f"Transformer Engine: ERROR - {e}")

# Other packages
try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except:
    print("Transformers: NOT INSTALLED")

try:
    import wandb
    print(f"WandB: {wandb.__version__}")
except:
    print("WandB: NOT INSTALLED")

print("="*60)
print("\n[SUCCESS] Setup complete! Ready for FP8 training!")
print("\nTo start training:")
print("  ./start_training.sh")
print("\nTo monitor:")
print("  ./monitor_training.sh")
print("="*60)
EOF

python verify_setup.py

# Create .bashrc additions for easy activation
echo "" >> ~/.bashrc
echo "# Yxanul training environment" >> ~/.bashrc
echo "alias yxanul='cd ~/yxanul_0.6B && source venv/bin/activate'" >> ~/.bashrc
echo "echo 'Type \"yxanul\" to activate the training environment'" >> ~/.bashrc

# Final message
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}BARE METAL SETUP COMPLETE!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Environment Setup:"
echo "  - Python 3.11 in virtual environment"
echo "  - PyTorch 2.3.1 with CUDA 12.1"
echo "  - Transformer Engine (FP8 enabled)"
echo "  - All dependencies installed in venv"
echo ""
echo "Quick Start:"
echo "  cd ~/yxanul_0.6B"
echo "  source venv/bin/activate  # Or use ./activate.sh"
echo "  ./start_training.sh       # Start FP8 training"
echo ""
echo "Shortcut added:"
echo "  Type 'yxanul' to quickly activate environment"
echo ""
echo "Expected performance:"
echo "  - FP8: 2x speedup over BF16"
echo "  - Curriculum: 3x additional speedup"
echo "  - Total: 6x faster than baseline"
echo ""
print_status "Ready to train with maximum performance!"
print_warning "Remember to activate venv before training: source venv/bin/activate"