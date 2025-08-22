# Curriculum Learning Optimization Plan

## Current Issues to Fix

1. **Batch size doesn't actually change** - DataLoader created once with fixed batch_size
2. **Sequence length stages too sparse** - Only 4 stages (256→512→1024→2048)
3. **Starting too high** - 256 tokens vs Microsoft's 8 tokens start
4. **Memory underutilized early** - Could process more samples when sequences are short

## Microsoft's DeepSpeed Results

- **3.3x faster pretraining** to target perplexity
- **Start: 8 tokens** → End: 1024 tokens
- **8x larger batch sizes** when sequences are short
- **49-70% wall-clock reduction**

## Proposed Optimized Curriculum

### Aggressive Curriculum (Microsoft-style)
```yaml
curriculum_stages:
  # Ultra-short: Learn token relationships
  - {step: 0,      seq_len: 64,   batch_size: 512}  # 32,768 tokens/batch
  - {step: 2000,   seq_len: 128,  batch_size: 256}  # 32,768 tokens/batch
  - {step: 5000,   seq_len: 256,  batch_size: 128}  # 32,768 tokens/batch
  
  # Short: Learn phrases and sentences  
  - {step: 10000,  seq_len: 512,  batch_size: 64}   # 32,768 tokens/batch
  
  # Medium: Learn paragraphs
  - {step: 20000,  seq_len: 1024, batch_size: 32}   # 32,768 tokens/batch
  
  # Long: Learn document structure (optional)
  - {step: 40000,  seq_len: 2048, batch_size: 16}   # 32,768 tokens/batch
```

### Key Principle: **Constant tokens per batch**
- All stages process ~32,768 tokens per batch
- GPU memory usage stays constant
- Early stages process MORE examples (512 examples of 64 tokens each!)

## 2048 vs 1024 Max Length Analysis

### 1024 Max Length (Microsoft/DeepSpeed)
**Pros:**
- ✅ 2x faster training per epoch
- ✅ More gradient updates per hour
- ✅ Sufficient for most text understanding
- ✅ Better for Q&A, chat, code (most are <1024)

**Cons:**
- ❌ Can't learn long document structure
- ❌ Loses context in long articles
- ❌ May struggle with complex reasoning chains

### 2048 Max Length (Current)
**Pros:**
- ✅ Learns document-level coherence
- ✅ Better for long-form generation
- ✅ Captures more Wikipedia article structure
- ✅ Better for scientific/technical content

**Cons:**
- ❌ 2x slower training
- ❌ Fewer examples seen per hour
- ❌ Diminishing returns (most benefit is <1024)

### Recommendation: **Hybrid Approach**
```yaml
# Stage 1: Fast learning with 1024 max (80% of training)
max_sequence_length: 1024
curriculum_stages:
  - {step: 0,     seq_len: 64,   batch_size: 256}
  - {step: 5000,  seq_len: 128,  batch_size: 128}
  - {step: 10000, seq_len: 256,  batch_size: 64}
  - {step: 20000, seq_len: 512,  batch_size: 32}
  - {step: 40000, seq_len: 1024, batch_size: 16}

# Stage 2: Fine-tune with 2048 (20% of training)
# Only after model has learned basics
```

## Implementation Requirements

### 1. Dynamic DataLoader Recreation
```python
def update_batch_size(step, curriculum_stages):
    for stage in curriculum_stages:
        if step >= stage['step']:
            current_batch_size = stage['batch_size']
            current_seq_len = stage['seq_len']
    
    # Recreate dataloader if batch_size changed
    if current_batch_size != last_batch_size:
        train_dataloader = create_dataloader(
            batch_size=current_batch_size,
            # ... other params
        )
```

### 2. Learning Rate Scaling
```python
# Scale LR with batch size (linear scaling rule)
base_lr = 6e-4
effective_batch_size = batch_size * seq_len
lr = base_lr * (effective_batch_size / 32768)  # Normalized to baseline
```

### 3. Warmup Adjustment
```python
# Shorter warmup for short sequences
warmup_steps = min(2000, step_to_next_stage * 0.1)
```

## Expected Benefits

### Training Speed
- **Early stages**: 8-16x more examples per hour
- **Overall**: 2-3x faster to target perplexity
- **First 10k steps**: See 640k examples instead of 40k

### Learning Quality
- **Better fundamentals**: Short sequences teach token relationships
- **Gradual complexity**: Natural progression in difficulty
- **Stable training**: Less likely to diverge early

### GPU Efficiency
- **Constant memory**: ~12GB throughout training
- **Higher throughput**: Better GPU utilization with larger batches
- **No OOM risk**: Memory usage controlled

## Metrics to Track

```python
metrics_to_log = {
    "curriculum/current_seq_len": current_seq_length,
    "curriculum/current_batch_size": current_batch_size,
    "curriculum/examples_per_step": current_batch_size,
    "curriculum/tokens_per_step": current_batch_size * current_seq_length,
    "curriculum/stage": current_stage_index,
}
```

## Quick Start for Next Run

1. **Use stage1_curriculum_optimized.yaml** (to be created)
2. **Enable dynamic batch size** in train.py
3. **Start with 64 tokens** not 256
4. **Target 1024 max** for first 100k steps
5. **Monitor examples/hour** not just tokens/second

## Key Insight

> "It's better to see 1 million simple examples quickly than 100k complex examples slowly"

The model needs to learn:
1. Token relationships (64 tokens)
2. Phrase structure (128 tokens)  
3. Sentence grammar (256 tokens)
4. Paragraph flow (512 tokens)
5. Document structure (1024+ tokens)

Starting with 2048 is like teaching calculus before arithmetic!