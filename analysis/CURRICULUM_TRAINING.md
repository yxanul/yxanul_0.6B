# Curriculum Training with Direct HuggingFace Streaming

## Overview
The curriculum training system streams directly from HuggingFace datasets without creating intermediate files. This saves disk space and eliminates preprocessing time.

## Architecture

### 1. Data Pipeline (`src/data_pipeline.py`)
- **CurriculumStreamingDataset**: Streams directly from HuggingFace with evolving dataset mix
- **create_curriculum_dataloader**: Creates curriculum-aware dataloaders
- Handles automatic stage transitions based on token count

### 2. Training Script (`train_te_v2.py`)
- Use `--curriculum` flag to enable curriculum training
- Automatically transitions through 10 stages
- Tracks cumulative tokens and stops at 1B tokens
- Adjusts batch sizes and sequence lengths per stage

### 3. Testing Utility (`test_curriculum_streaming.py`)
- Verifies streaming functionality
- Tests dataset mix ratios
- Estimates training time
- Checks stage transitions

## Dataset Sources (All Streaming)

1. **FineWeb-Edu**: `HuggingFaceFW/fineweb-edu` (10B tokens available)
2. **Math**: `Yxanul/cc-math-finest` (high-quality mathematical text)
3. **Code**: `Yxanul/python-finest-pretrain` (Python code)

## Usage

### Start Curriculum Training (1B tokens)
```bash
# With TE v2.4 (recommended for RTX 4090/H100)
python train_te_v2.py --curriculum --model-size 270M --target-tokens 1000000000

# Specific curriculum config
python train_te_v2.py --curriculum --curriculum-config configs/yxanul_270m_progressive_curriculum.yaml
```

### Test Streaming Before Training
```bash
# Test first and last stages
python test_curriculum_streaming.py

# Test all 10 stages
python test_curriculum_streaming.py --test-all

# Test specific stage (e.g., stage 5)
python test_curriculum_streaming.py --stage 5

# Test stage transitions
python test_curriculum_streaming.py --test-transitions
```

## Curriculum Stages

| Stage | Name | Seq Len | Tokens | FineWeb | Math | Code |
|-------|------|---------|--------|---------|------|------|
| 1 | Basic Patterns | 8 | 10M | 95% | 5% | 0% |
| 2 | Sentences | 16 | 20M | 95% | 5% | 0% |
| 3 | Paragraphs | 32 | 30M | 90% | 8% | 2% |
| 4 | Extended Context | 64 | 40M | 85% | 10% | 5% |
| 5 | Reasoning Development | 128 | 75M | 80% | 12% | 8% |
| 6 | Complex Patterns | 256 | 100M | 75% | 15% | 10% |
| 7 | Integration | 512 | 125M | 70% | 18% | 12% |
| 8 | Deep Understanding | 768 | 150M | 65% | 20% | 15% |
| 9 | Expert Reasoning | 1536 | 200M | 60% | 22% | 18% |
| 10 | Final Polish | 2048 | 250M | 55% | 25% | 20% |

Total: 1B tokens with progressive complexity and evolving dataset mix.

## Key Features

1. **Direct Streaming**: No intermediate JSONL files needed
2. **Automatic Stage Transitions**: Based on token count
3. **Dynamic Batch Sizes**: Adjusted per stage for GPU efficiency
4. **Evolving Dataset Mix**: Gradually increases math/code content
5. **Token Tracking**: Automatically stops at target (1B tokens)

## Expected Performance

On RTX 4090 with TE v2.4:
- Stage 1-4 (short sequences): ~150k tokens/sec
- Stage 5-7 (medium sequences): ~100k tokens/sec  
- Stage 8-10 (long sequences): ~50k tokens/sec
- **Total time**: ~4-6 hours for 1B tokens

## Implementation Details

The system uses:
- **Streaming iterators** from HuggingFace datasets library
- **Probabilistic sampling** to maintain dataset mix ratios
- **Buffer management** to prevent memory overflow
- **Dynamic stride** (50% overlap for efficiency)
- **EOS tokens** between documents for proper boundaries

## Monitoring Progress

During training, you'll see:
```
Step 1000: Loss=4.5421, PPL=93.84, Tokens/sec=85432, LR=6.00e-04 | Stage 3/10 (30,000,000/1,000,000,000 tokens, 3.0%)
```

## Notes

- Streaming requires stable internet connection
- First batch may take longer due to dataset initialization
- Use `test_curriculum_streaming.py` to verify connectivity before training
- Batch sizes are automatically adjusted to be multiples of 8 for FP8 efficiency