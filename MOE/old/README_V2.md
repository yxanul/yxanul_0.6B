# MoE V2 Implementation

## Overview

This is the optimized V2 implementation of our Mixture of Experts model, incorporating all performance improvements discovered during systematic profiling and optimization.

## Key Architecture Changes from V1

### 1. **No Base MLP** ✅ 2.06x Speedup
- **V1**: Always-active base MLP + expert MLP = 2x compute
- **V2**: Only expert MLP active = optimal compute
- **Why**: With k=1 (single expert), base MLP is redundant

### 2. **Switch Transformer Routing** ✅ Balanced Load
- **V1**: Sigmoid routing with variable activation
- **V2**: Switch routing with capacity limits
- **Why**: Guarantees balanced GPU utilization

### 3. **Larger Experts** ✅ Better Capacity
- **V1**: 1.25-1.75x expansion
- **V2**: 3.5-4.0x expansion  
- **Why**: Can afford larger experts without base overhead

### 4. **Native GQA Support** ✅ 4x Memory Savings
- **V1**: repeat_interleave causing 4x memory blow-up
- **V2**: Flash Attention with native GQA
- **Why**: Efficient memory usage for grouped queries

## Performance Improvements

| Metric | V1 (PR-MoE) | V2 (Switch) | Improvement |
|--------|-------------|-------------|-------------|
| Forward Pass | 159ms | 77ms | 2.06x |
| Throughput | 57k tok/s | 150k+ tok/s | 2.6x |
| Memory Usage | High | Optimal | -60% |
| Expert Balance (CV) | 0.73 | <0.3 | 2.4x better |

## Configuration

### Recommended Settings

```python
config = MoEConfig(
    # Model dimensions
    n_embd=896,
    n_head=28,
    n_kv_head=7,  # GQA with 4x reduction
    n_layer=24,
    
    # MoE settings
    num_experts=4,  # 2-4 experts optimal
    expert_expansion=3.5,  # Larger since no base
    capacity_factor=1.0,  # Start conservative
    
    # Routing
    router_type="switch",
    router_aux_loss_weight=0.01,
    overflow_policy="drop",  # or "rescue"
)
```

### Capacity Factor Tuning

- `c=1.0`: Conservative, some tokens dropped, fastest
- `c=1.25`: Balanced, minimal drops, good utilization  
- `c=1.5`: Liberal, almost no drops, slightly slower

### Overflow Policies

- `"drop"`: Drop excess tokens (default, fastest)
- `"rescue"`: Route overflow to small rescue FFN (more stable)

## Training

### Quick Start

```bash
# Standard training
python train_moe_v2.py \
    --batch_size 12 \
    --grad_accum_steps 1 \
    --num_experts 4 \
    --capacity_factor 1.25

# With mixed precision
python train_moe_v2.py \
    --batch_size 16 \
    --use_amp \
    --capacity_factor 1.0

# With W&B logging
python train_moe_v2.py \
    --wandb \
    --wandb_project "moe-v2"
```

### Optimal Batch Sizes (32GB VRAM)

| Experts | Expansion | Batch Size | Grad Accum |
|---------|-----------|------------|------------|
| 2 | 4.0x | 16 | 1 |
| 4 | 3.5x | 12 | 1 |
| 4 | 3.5x | 8 | 2 |
| 6 | 3.0x | 8 | 2 |

## Benchmarking

```bash
# Run comprehensive benchmarks
python benchmark_v2.py

# Expected output:
# - 150k+ tokens/sec throughput
# - <80ms forward pass
# - CV < 0.3 (good balance)
# - Drop rate < 5% (with c=1.25)
```

## Architecture Details

### Switch Routing Algorithm

```python
1. Compute router scores: softmax(router(x))
2. Select top-1 expert per token
3. Apply capacity limit: C = ceil(N * c / E)
4. Keep top-C tokens per expert (by score)
5. Drop overflow tokens (or send to rescue)
6. Process batches through experts
```

### Load Balancing

The model uses two complementary losses:

1. **Auxiliary Loss**: Encourages uniform distribution
   - `loss = sum(f_i * P_i) * E`
   - `f_i` = fraction dispatched to expert i
   - `P_i` = probability mass to expert i

2. **Z-Loss**: Prevents router logit explosion
   - `loss = log(sum(exp(logits)))^2`

### Expert Execution Flow

```python
for each expert:
    1. Gather assigned tokens (efficient indexing)
    2. Pad to multiple of 16 (cuBLAS alignment)
    3. Process through expert FFN (SwiGLU)
    4. Weight by routing probability
    5. Scatter back to sequence
```

## Monitoring

The model provides comprehensive statistics:

- **Expert loads**: Token distribution across experts
- **CV (Coefficient of Variation)**: Load balance metric (target < 0.3)
- **Drop rate**: Fraction of tokens dropped due to capacity
- **Router loss**: Combined auxiliary + z-loss

These are logged every `log_interval` steps during training.

## Key Insights from Optimization

1. **Architecture > Micro-optimizations**: Removing base MLP gave 2x speedup
2. **Hard constraints work**: Capacity limits prevent expert collapse
3. **Monitor everything**: Silent expert collapse is common
4. **Padding matters**: 16-token alignment critical for FP8/cuBLAS
5. **Index operations > masked ops**: 3-5x faster token gathering

## Files

- `model_moe_v2.py`: Core V2 model implementation
- `train_moe_v2.py`: Optimized training script
- `benchmark_v2.py`: Performance validation
- `OPTIMIZATION_FINDINGS.md`: Detailed optimization journey

## Citation

Based on Switch Transformer architecture:
```
@article{fedus2022switch,
  title={Switch Transformers: Scaling to Trillion Parameter Models},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={JMLR},
  year={2022}
}
```

## Next Steps

- [ ] Test with 6-8 experts for larger models
- [ ] Implement expert choice routing as alternative
- [ ] Add MTP back as optional feature
- [ ] Test INT8 quantization for inference
- [ ] Implement pipeline parallelism for multi-GPU