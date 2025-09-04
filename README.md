# 🚀 Elite-112M: Ultra-Efficient Language Model Experiment

*A proof-of-concept demonstrating that meaningful language model behavior can emerge with just 3 hours of training on consumer hardware*

<div align="center">

![Model Size](https://img.shields.io/badge/Parameters-112M-blue)
![Training Time](https://img.shields.io/badge/Training%20Time-3%20hours-green)
![Hardware](https://img.shields.io/badge/Hardware-RTX%205090-red)
![Tokens](https://img.shields.io/badge/Pretrain%20Tokens-2.5B-orange)
![Speed](https://img.shields.io/badge/Speed-196k%20tok/s-purple)

</div>

## 📖 Table of Contents
- [Overview](#overview)
- [Key Achievements](#key-achievements)
- [Architecture](#architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Insights](#technical-insights)
- [Limitations](#limitations)
- [Future Work](#future-work)

## 🎯 Overview

This project demonstrates that Language Model research doesn't require massive compute budgets. We trained a 112M parameter GPT model from scratch in just **3 hours** on a single **RTX 5090**, achieving:
- **2.2 validation loss** after pretraining on 2.5B tokens
- **2.06 SFT loss** after fine-tuning on 60k instruction examples
- **196k tokens/second** training throughput using FP8 precision

The model successfully learned conversation formats, code structure patterns, and basic factual knowledge - proving that rapid experimentation is possible on consumer hardware.

## 🏆 Key Achievements

### Speed Records
- **196k tokens/sec** with FP8 precision (CLEAN implementation)
- **185k tokens/sec** with "optimized" version (slower due to overhead)
- **15-25 tokens/sec** inference on CPU

### Learning Milestones
✅ **Format Learning**
- User/Assistant conversation structure
- GSM8K math answer format (`<<computation>>` brackets)
- Python code block formatting
- Markdown structure

✅ **Pattern Recognition**
- Number sequences (2, 4, 6, 8 → 10)
- Basic comparisons (elephant > mouse)
- Code syntax patterns

⚠️ **Approximate Learning**
- Math computations (7+8=14, close but wrong)
- Partial instruction following
- Some factual knowledge (Paris is capital of France)

## 🏗️ Architecture

### Model Configuration
```python
ModelConfig:
  - Parameters: 112.1M
  - Layers: 12
  - Attention Heads: 12
  - Embedding Dimension: 768
  - KV Heads: 3 (GQA with 4x compression)
  - Context Length: 2048
  - Vocabulary Size: 49,152 (SmolLM2 tokenizer)
```

### Key Innovations: CLEAN Architecture

The "CLEAN" implementation (`model_te_clean.py`) removes "optimizations" that actually slow down small models:

1. **NO FP8 weight caching** - Overhead exceeds benefit at 112M scale
2. **NO gradient accumulation fusion** - Adds memory traffic without speedup
3. **NO complex optimizations** - Simple is faster for small models
4. **Removed dead code** - No `is_first_microbatch` parameter threading

```python
# CLEAN approach - actually faster!
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    logits, loss = model(x, y)  # No is_first_microbatch!
```

### Components

- **Attention**: Multi-Head Attention with GQA (Grouped Query Attention)
- **FFN**: SwiGLU feedforward network
- **Positional Encoding**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMSNorm
- **Precision**: Mixed FP8/BF16 training with TransformerEngine

## 📊 Training Process

### Phase 1: Pretraining (3 hours)
- **Dataset**: 2.5B tokens from mixed sources
- **Batch Size**: 8 with gradient accumulation (effective: 128)
- **Learning Rate**: 8e-4 with cosine decay
- **Hardware**: Single RTX 5090
- **Result**: 2.2 validation loss

### Phase 2: SFT (Supervised Fine-Tuning)
- **Dataset**: 60k examples (Alpaca + GSM8K format)
- **Learning Rate**: 3e-4 (aggressive for quick adaptation)
- **Iterations**: 50 (very short!)
- **Result**: 2.06 validation loss

### Training Efficiency
```
Tokens per second: 196,000
Total tokens seen: 2.5B
Time to 1 epoch: ~3 hours
GPU Memory: <24GB
Power consumption: ~450W
```

## 📈 Results

### Evaluation Metrics

| Category | Success Rate | Example |
|----------|-------------|---------|
| **Factual Q&A** | 20% | ✅ "Capital of France" → "Paris" |
| **Math Format** | 90% | ✅ Uses `<<>>` brackets correctly |
| **Math Accuracy** | 10% | ❌ 7+8=14 (wrong but close!) |
| **Code Structure** | 70% | ✅ Valid Python syntax |
| **Instructions** | 30% | ⚠️ Mixed success |
| **Pattern Recognition** | 60% | ✅ 2,4,6,8 → 10 |

### Sample Outputs

**Success Case:**
```
User: What is the capital of France?
Assistant: The capital of France is Paris.
```

**Format Learning (Math):**
```
User: What is 7 + 8?
Assistant: 7 + 8 = <<7+8=14>>14  # Wrong answer, correct format!
```

**Code Generation:**
```python
User: Write a Python function to add two numbers.
Assistant: 
def add_numbers(num1, num2):
    return num1 + num2  # Structure correct!
```

## 📁 Project Structure

```
yxanul_0.6B/
│
├── train/                      # Pretraining components
│   ├── model_te_clean.py       # CLEAN model architecture (fastest)
│   ├── train_fp8_optimized.py  # Main training script with FP8
│   ├── prepare_fineweb_edu_quick.py  # Data preparation
│   └── wandb_logger.py         # Weights & Biases logging
│
├── sft/                        # Fine-tuning components
│   ├── train_sft.py            # SFT training script
│   ├── test_sft_model.py      # Comprehensive evaluation suite
│   ├── model_inference.py     # Inference-optimized model
│   ├── prepare_sft_data_parquet_fast.py  # SFT data prep
│   ├── best_sft_model.pt      # Final trained model
│   └── logs.txt                # Evaluation results
│
├── data_mixed_3b/              # Training data (generated)
│   ├── train.bin               # Binary training tokens
│   └── val.bin                 # Binary validation tokens
│
└── checkpoints_*/              # Model checkpoints
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `model_te_clean.py` | Core model architecture with CLEAN optimizations |
| `train_fp8_optimized.py` | Pretraining script with FP8 precision |
| `train_sft.py` | Instruction fine-tuning implementation |
| `test_sft_model.py` | Evaluation suite with 35+ test categories |
| `wandb_logger.py` | Robust logging with automatic recovery |
| `model_inference.py` | CPU-optimized inference version |

## 🚀 Installation

### Prerequisites
- NVIDIA GPU with FP8 support (RTX 4090/5090 or H100)
- CUDA 12.1+
- Python 3.10+

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/yxanul_0.6B.git
cd yxanul_0.6B

# Install dependencies
pip install torch transformers
pip install transformer-engine  # For FP8 support
pip install numpy pandas tqdm
pip install wandb  # Optional: for logging

# Download tokenizer
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')"
```

## 💻 Usage

### Quick Start - Test the Model
```bash
# Run comprehensive evaluation
python sft/test_sft_model.py --mode test

# Interactive chat
python sft/test_sft_model.py --mode chat

# Single query
python sft/test_sft_model.py --mode single --input "What is 2+2?"
```

### Training from Scratch

#### 1. Prepare Data
```bash
# Prepare pretraining data (2.5B tokens)
python train/prepare_fineweb_edu_quick.py

# Prepare SFT data (60k examples)
python sft/prepare_sft_data_parquet_fast.py
```

#### 2. Pretrain Model
```bash
# Train with FP8 (fastest - 196k tok/s)
python train/train_fp8_optimized.py \
    --batch_size 8 \
    --max_iters 2000 \
    --eval_interval 100

# Train without FP8 (slower but more compatible)
python train/train_fp8_optimized.py --no_fp8
```

#### 3. Fine-tune for Instructions
```bash
# Run SFT on pretrained model
python sft/train_sft.py \
    --base_model best_model_fp8_optimized.pt \
    --max_iters 50 \
    --learning_rate 3e-4
```

### Inference Options

```python
# CPU Inference (no GPU required)
from sft.model_inference import GPTInference, ModelConfig

config = ModelConfig(n_layer=12, n_head=12, n_embd=768)
model = GPTInference(config)
model.load_from_te_checkpoint('sft/best_sft_model.pt')

# Generate text
output = model.generate(prompt, max_tokens=50, temperature=0.7)
```

## 🔬 Technical Insights

### Why This Works

1. **Format Over Function**: Small models excel at pattern matching
   - Learned conversation structure perfectly
   - Memorized code syntax patterns
   - Captured mathematical notation formats

2. **Efficiency Wins**: CLEAN architecture proves simpler is faster
   - Removed FP8 weight caching (overhead > benefit)
   - Eliminated gradient fusion (memory traffic)
   - Stripped complex optimizations

3. **Aggressive Training**: High learning rates work at small scale
   - 8e-4 for pretraining (10x typical)
   - 3e-4 for SFT (very aggressive)
   - Minimal warmup (5 iterations)

### Key Discoveries

- **Emergent Abilities at 112M**:
  - Basic pattern recognition
  - Format memorization
  - Approximate arithmetic
  
- **Scaling Insights**:
  - 2.5B tokens sufficient for format learning
  - 60k SFT examples enough for conversation structure
  - 3 hours achieves meaningful behavior

### Optimization Techniques

```python
# Greedy decoding for deterministic outputs
temperature = 0.0  # No randomness
top_k = 0         # Argmax selection
max_tokens = 16   # Prevent rambling

# Memory-efficient data loading
data = np.memmap('train.bin', dtype=np.uint16, mode='r')

# FP8 training recipe
recipe = DelayedScaling(
    fp8_format=Format.HYBRID,  # E4M3 forward, E5M2 backward
    amax_history_len=16
)
```

## ⚠️ Limitations

### Current Limitations
- **Factual Accuracy**: ~20% correct on knowledge questions
- **Math Computation**: Learned format but not arithmetic
- **Instruction Following**: Inconsistent compliance
- **Context Length**: Limited to 2048 tokens
- **Vocabulary**: English-only with 49k tokens

### Not Suitable For
- Production deployments
- Factual question answering
- Mathematical calculations
- Complex reasoning tasks
- Multi-turn conversations

## 🔮 Future Work

### Immediate Improvements
1. **Scale to 350M parameters** - Cross the reasoning threshold
2. **Train for 10B tokens** - Improve knowledge retention
3. **Arithmetic dataset** - Fix math computation
4. **Instruction diversity** - Better compliance

### Research Directions
- Knowledge distillation from larger models
- Mixture of Experts (MoE) at small scale
- Specialized tokenizers for efficiency
- Continued pretraining techniques

### Community Contributions Welcome
- [ ] ONNX export for edge deployment
- [ ] Quantization to 4-bit
- [ ] WebGPU implementation
- [ ] Mobile deployment
- [ ] Domain-specific fine-tuning

## 📝 Citation

If you find this work useful, please cite:
```bibtex
@misc{elite112m2024,
  title={Elite-112M: Ultra-Efficient Language Model Training on Consumer Hardware},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/yxanul_0.6B}
}
```

## 🙏 Acknowledgments

- **TransformerEngine** team for FP8 implementation
- **HuggingFace** for SmolLM tokenizer
- **OpenAI** for GPT architecture inspiration
- The open-source ML community

## 📜 License

MIT License - See LICENSE file for details

---

<div align="center">

**Remember: Meaningful ML research doesn't require massive compute budgets.**

*Just clever engineering and clear thinking about what matters at each scale.*

🚀 **Happy experimenting!** 🚀

</div>