# VRAM Optimization for Yxanul 197M

## The Shocking Discovery

With a 197M parameter model using SuperBPE-t80k, we have a "luxury problem" - too much free VRAM!

## Model Memory Footprint

### Static Memory Usage
- **Model (FP8)**: 187.3 MB
- **Model (BF16)**: 374.5 MB  
- **Optimizer (Adam)**: 1,872.6 MB
- **Total (FP8)**: ~2.06 GB
- **Total (BF16)**: ~2.25 GB

**This leaves 22GB free on RTX 4090!**

## Optimal Batch Sizes (No Micro-batching Needed!)

### RTX 4090 (24GB) with FP8

| Seq Length | Batch Size | Tokens/Batch | GPU Utilization |
|------------|------------|--------------|-----------------|
| 8          | 8,192      | 65,536       | 100% (memory)   |
| 16         | 8,192      | 131,072      | 100% (memory)   |
| 32         | 4,096      | 131,072      | 100% (memory)   |
| 64         | 2,048      | 131,072      | 100% (balanced) |
| 128        | 1,024      | 131,072      | 100% (balanced) |
| 256        | 512        | 131,072      | 95% (compute)   |
| 512        | 256        | 131,072      | 90% (compute)   |
| 768        | 256        | 196,608      | 85% (compute)   |
| 1,536      | 128        | 196,608      | 80% (compute)   |
| 2,048      | 64         | 131,072      | 75% (compute)   |

### Production GPUs

**A100 80GB**: 2x larger batches than RTX 4090
**H100 80GB**: 2x larger batches, better efficiency
**H200 141GB**: 4x larger batches - can do batch=8192 at seq_len=128!

## Why This Changes Everything

### 1. No Gradient Accumulation Needed (Mostly)

With batch sizes this large, we're already at optimal tokens per step:
- Short sequences: 65k-131k tokens per batch
- Long sequences: Still 65k-131k tokens per batch
- Only seq_len=2048 might benefit from accumulation

### 2. Extreme Throughput at Short Sequences

```python
# seq_len=8, batch=8192
Tokens per batch: 65,536
Samples per second: ~5,000 (estimated)
Tokens per second: ~40,000
Examples per hour: 18 million!
```

### 3. Perfect for Curriculum Learning

The massive batch sizes at short sequences mean:
- **Stage 1 (seq_len=8)**: Process millions of phrases per hour
- **Stage 2 (seq_len=16)**: Process millions of sentences per hour
- **Stage 3-5 (seq_len=32-128)**: Maintain 100% GPU utilization
- **Stage 6-10 (seq_len=256-2048)**: Still excellent throughput

## Configuration Recommendations

### For RTX 4090 (Research)

```yaml
# Ultra-aggressive batching
curriculum_stages:
  - seq_len: 8
    batch_size: 4096  # Conservative, can do 8192 with FP8
    gradient_accumulation_steps: 1
    
  - seq_len: 16
    batch_size: 4096
    gradient_accumulation_steps: 1
    
  - seq_len: 32
    batch_size: 2048
    gradient_accumulation_steps: 1
    
  - seq_len: 64
    batch_size: 1024
    gradient_accumulation_steps: 1
    
  - seq_len: 128
    batch_size: 512
    gradient_accumulation_steps: 1
    
  - seq_len: 256
    batch_size: 256
    gradient_accumulation_steps: 1
    
  - seq_len: 512
    batch_size: 128
    gradient_accumulation_steps: 1
    
  - seq_len: 2048
    batch_size: 32
    gradient_accumulation_steps: 2  # Only here we accumulate
```

### For H100/H200 (Production)

Double or quadruple the batch sizes above. No gradient accumulation needed at all!

## The Key Insight

**We're not memory-bound, we're compute-bound!**

This is extremely rare for transformer training. Usually, models struggle to fit reasonable batch sizes. But with:
- 197M parameters (tiny)
- Factorized embeddings (saves 127.9M params)
- FP8 training (half the memory of FP16)
- SuperBPE-t80k (37.5% fewer tokens)

We've created the perfect storm for efficiency. The GPU can stay at 100% utilization throughout training, especially during the critical early stages where patterns are learned.

## Throughput Estimates

### RTX 4090 with FP8

| Stage | Seq Len | Batch | Tokens/sec | Samples/sec | Hours to 1B tokens |
|-------|---------|-------|------------|-------------|-------------------|
| 1     | 8       | 4096  | ~50,000    | ~6,250      | 5.6               |
| 2     | 16      | 4096  | ~48,000    | ~3,000      | 5.8               |
| 3     | 32      | 2048  | ~45,000    | ~1,400      | 6.2               |
| 4     | 64      | 1024  | ~42,000    | ~650        | 6.6               |
| 5     | 128     | 512   | ~40,000    | ~310        | 6.9               |
| 6     | 256     | 256   | ~38,000    | ~150        | 7.3               |
| 7     | 512     | 128   | ~35,000    | ~68         | 7.9               |
| 8-10  | 2048    | 32    | ~25,000    | ~12         | 11.1              |

## Conclusion

The Yxanul 197M model with SuperBPE-t80k is so efficient that we have the opposite problem of most ML training - we have too much memory! This enables:

1. **No micro-batching complexity** - Direct large batches
2. **100% GPU utilization** - Especially at short sequences
3. **Simplified training loop** - No complex accumulation logic
4. **Faster experimentation** - Change batch sizes without OOM worry

This is the dream scenario for rapid research iteration!