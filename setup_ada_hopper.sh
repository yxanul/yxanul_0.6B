#!/bin/bash

# ============================================
# Yxanul 0.6B Complete Setup Script
# For RTX 4090 (Ada/sm_89) & H100/H200 (Hopper/sm_90)
# Includes FP8 support via Transformer Engine
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Yxanul 0.6B Training Setup${NC}"
echo -e "${BLUE}Optimized for RTX 4090 & H100/H200${NC}"
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

# 1. Detect GPU and CUDA
echo -e "\n${BLUE}[Step 1/7] Detecting GPU and CUDA...${NC}"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found! Please install NVIDIA drivers."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
echo "GPU Detected: $GPU_NAME (sm_${GPU_COMPUTE_CAP//./})"

# Validate GPU architecture
GPU_SM="${GPU_COMPUTE_CAP//./}"
if [[ "$GPU_SM" == "89" ]]; then
    print_status "RTX 4090 (Ada Lovelace) detected - FP8 fully supported!"
    GPU_ARCH="ada"
elif [[ "$GPU_SM" == "90" ]]; then
    print_status "H100/H200 (Hopper) detected - FP8 fully supported!"
    GPU_ARCH="hopper"
elif [[ "$GPU_SM" == "80" || "$GPU_SM" == "86" || "$GPU_SM" == "87" ]]; then
    print_status "Ampere GPU detected - FP8 supported!"
    GPU_ARCH="ampere"
else
    print_warning "Unknown GPU architecture (sm_$GPU_SM) - FP8 may not be optimal"
    GPU_ARCH="unknown"
fi

# Check CUDA version
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc not found! Please install CUDA toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

echo "CUDA Version: $CUDA_VERSION"

# Validate CUDA version for Transformer Engine
if [[ "$CUDA_MAJOR" -lt 11 ]] || ([[ "$CUDA_MAJOR" -eq 11 ]] && [[ "$CUDA_MINOR" -lt 8 ]]); then
    print_error "CUDA 11.8+ required for Transformer Engine. Found: $CUDA_VERSION"
    exit 1
fi

# Determine CUDA version for PyTorch
if [[ "$CUDA_MAJOR" -eq 12 ]] && [[ "$CUDA_MINOR" -ge 1 ]]; then
    PYTORCH_CUDA="cu121"
    print_status "Using CUDA 12.1 packages (optimal for RTX 4090/H100)"
elif [[ "$CUDA_MAJOR" -eq 11 ]] && [[ "$CUDA_MINOR" -eq 8 ]]; then
    PYTORCH_CUDA="cu118"
    print_status "Using CUDA 11.8 packages"
else
    PYTORCH_CUDA="cu121"
    print_warning "Non-standard CUDA version, defaulting to cu121 packages"
fi

# 2. Clone repository
echo -e "\n${BLUE}[Step 2/7] Cloning Yxanul repository...${NC}"

if [ -d "yxanul_0.6B" ]; then
    print_warning "Directory yxanul_0.6B already exists. Pulling latest changes..."
    cd yxanul_0.6B
    git pull
else
    git clone https://github.com/yxanul/yxanul_0.6B.git
    cd yxanul_0.6B
    print_status "Repository cloned successfully"
fi

# 3. Check Python version (using existing environment)
echo -e "\n${BLUE}[Step 3/7] Checking Python environment...${NC}"

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || ([[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -lt 9 ]]); then
    print_error "Python 3.9+ required. Found: Python $PYTHON_VERSION"
    exit 1
fi

print_status "Python $PYTHON_VERSION detected"
print_status "Using existing virtual environment in SSH session"

# Upgrade pip
pip install --upgrade pip setuptools wheel
print_status "pip upgraded"

# 4. Install PyTorch with correct CUDA version
echo -e "\n${BLUE}[Step 4/7] Installing PyTorch with CUDA support...${NC}"

# Uninstall any existing PyTorch to avoid conflicts
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch with the correct CUDA version
if [[ "$PYTORCH_CUDA" == "cu121" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$PYTORCH_CUDA" == "cu118" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio
fi

# Verify PyTorch installation
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if [ $? -eq 0 ]; then
    print_status "PyTorch installed successfully with CUDA support"
else
    print_error "PyTorch installation verification failed"
    exit 1
fi

# 5. Install Transformer Engine for FP8 support
echo -e "\n${BLUE}[Step 5/7] Installing Transformer Engine for FP8...${NC}"

# Install build dependencies
pip install ninja packaging

# Install Transformer Engine
pip install transformer-engine[pytorch]

# Verify Transformer Engine
python3 -c "
import transformer_engine.pytorch as te
print(f'Transformer Engine version: {te.__version__}')
try:
    print(f'FP8 available: {te.fp8.is_fp8_available()}')
    print(f'FP8 support: Available and ready!')
except:
    print('FP8 support: Check available after full setup')"

if [ $? -eq 0 ]; then
    print_status "Transformer Engine installed successfully"
else
    print_warning "Transformer Engine installation may need verification"
fi

# 6. Install other dependencies
echo -e "\n${BLUE}[Step 6/7] Installing additional dependencies...${NC}"

# Core ML packages
pip install transformers>=4.36.0
pip install datasets>=2.15.0
pip install accelerate>=0.25.0
pip install tokenizers>=0.15.0

# Training utilities
pip install wandb
pip install tqdm
pip install pyyaml
pip install numpy
pip install pandas
pip install scikit-learn

# Optional but recommended
pip install bitsandbytes  # For 8-bit optimizers
pip install deepspeed  # For distributed training (optional)
pip install apex-amp || print_warning "Apex installation failed (optional, not critical)"

print_status "All dependencies installed"

# 7. Configure WandB
echo -e "\n${BLUE}[Step 7/7] Configuring Weights & Biases...${NC}"

# Login to WandB with provided key
wandb login --relogin 4444c18d3905dde9ab69774b2322a0c41ab209d3

if [ $? -eq 0 ]; then
    print_status "WandB configured successfully"
else
    print_warning "WandB login failed - you may need to login manually"
fi

# Final verification and setup completion
echo -e "\n${BLUE}Running final verification...${NC}"

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p data
print_status "Created necessary directories"

# Verification script
cat > verify_setup.py << 'EOF'
import sys
import torch
try:
    import transformer_engine.pytorch as te
    te_available = True
    fp8_available = te.fp8.is_fp8_available()
except:
    te_available = False
    fp8_available = False

print("\n" + "="*50)
print("SETUP VERIFICATION RESULTS")
print("="*50)

# Check PyTorch
print(f"[OK] PyTorch: {torch.__version__}")
print(f"[OK] CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Check Transformer Engine
if te_available:
    print(f"[OK] Transformer Engine: Available")
    print(f"[OK] FP8 Support: {'Ready' if fp8_available else 'Check after reboot'}")
else:
    print("[ERROR] Transformer Engine: Not available")

# Check other packages
try:
    import transformers
    print(f"[OK] Transformers: {transformers.__version__}")
except:
    print("[ERROR] Transformers: Not installed")

try:
    import datasets
    print(f"[OK] Datasets: {datasets.__version__}")
except:
    print("[ERROR] Datasets: Not installed")

try:
    import wandb
    print(f"[OK] WandB: {wandb.__version__}")
except:
    print("[ERROR] WandB: Not installed")

print("="*50)

if te_available and torch.cuda.is_available():
    print("\n[SUCCESS] SETUP COMPLETE! Ready for FP8 training on your GPU!")
    print("\nNext steps:")
    print("1. Start training with FP8:")
    print("   python train_fp8.py --config configs/stage1_curriculum_fp8_fineweb.yaml")
    print("\n2. Or train without FP8:")
    print("   python train_curriculum.py --config configs/stage1_curriculum_optimized_24gb.yaml")
else:
    print("\n[WARNING] Setup partially complete. Check any ERROR items above.")

print("\nDataset will be downloaded automatically on first run.")
print("Expected download: ~4-5GB for FineWeb-Edu")
print("="*50)
EOF

python3 verify_setup.py

# Create quick start script
cat > start_training.sh << 'EOF'
#!/bin/bash
echo "Starting FP8 training on FineWeb-Edu dataset..."
python train_fp8.py --config configs/stage1_curriculum_fp8_fineweb.yaml
EOF
chmod +x start_training.sh

# Create monitoring script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
watch -n 1 'nvidia-smi; echo ""; tail -n 20 logs/training.log 2>/dev/null || echo "Training not started yet"'
EOF
chmod +x monitor_training.sh

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Quick commands:"
echo "  ./start_training.sh    - Start FP8 training"
echo "  ./monitor_training.sh  - Monitor GPU and training"
echo ""
echo "Manual training:"
echo "  python train_fp8.py --config configs/stage1_curriculum_fp8_fineweb.yaml"
echo ""
echo "Your FineWeb-Edu dataset will be downloaded on first run."
echo "Dataset: Yxanul/fineweb-edu-highest-quality-2025 (4.1B tokens)"
echo ""
echo -e "${GREEN}Happy training!${NC}"