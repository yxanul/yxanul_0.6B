# Full Epoch SFT Training Strategy

## Why Full Epoch Training Makes Sense Here

With 75k examples (12.6M tokens), we can afford to do proper epoch-based training rather than arbitrary iteration counts.

## Configuration Changes

### Previous (Large Model Approach)
- **Batch**: 16
- **Gradient Accumulation**: 16  
- **Effective Batch**: 256 examples
- **Updates per Epoch**: 24
- **Max Iters**: 3000 (overkill)

### New (Small Model Approach)
- **Batch**: 8
- **Gradient Accumulation**: 1
- **Effective Batch**: 8 examples  
- **Updates per Epoch**: 772
- **Max Iters**: 800 (1 full epoch)

## Benefits of This Approach

### 1. More Frequent Updates
- **Every 8 examples** instead of every 256
- Better gradient signal for small model
- Faster adaptation to instruction format
- Less memory usage (no gradient accumulation)

### 2. Natural Training Progression
```
Iterations 0-20:    Warmup (2.5% of epoch)
Iterations 20-200:  Rapid format learning (25% of epoch)  
Iterations 200-400: Pattern refinement (50% of epoch)
Iterations 400-600: Fine details (75% of epoch)
Iterations 600-772: Final polish (100% of epoch)
```

### 3. Better Learning Dynamics
- Each example seen exactly once
- No arbitrary cutoffs
- Natural curriculum from easy to hard
- Evaluation every 50 iterations (~6% of data)

## Exact Math

```python
Total tokens: 12,645,675
Batch size: 8
Block size: 2048
Tokens per batch: 8 × 2048 = 16,384

Iterations per epoch: 12,645,675 ÷ 16,384 = 772 iterations
```

## Recommended Training Commands

### Standard 1 Epoch
```bash
python train_sft.py \
    --batch_size 8 \
    --grad_accum 1 \
    --max_iters 800 \
    --eval_interval 50 \
    --learning_rate 7e-5
```

### Conservative 0.5 Epoch
```bash
python train_sft.py \
    --batch_size 8 \
    --grad_accum 1 \
    --max_iters 400 \
    --eval_interval 50 \
    --learning_rate 7e-5
```

### Thorough 2 Epochs
```bash
python train_sft.py \
    --batch_size 8 \
    --grad_accum 1 \
    --max_iters 1600 \
    --eval_interval 100 \
    --learning_rate 7e-5
```

## Learning Rate Schedule

With 800 iterations:
- **Warmup**: 20 iterations (2.5%)
- **Plateau**: 400 iterations (50%)  
- **Cosine Decay**: 380 iterations (47.5%)
- **Final LR**: 1e-5

## Expected Training Time

At ~200k tokens/sec with FP8:
- Tokens per second: 200,000
- Tokens per iteration: 16,384
- Time per iteration: 0.08 seconds
- **Total time for 1 epoch: ~65 seconds!**

## Monitoring Tips

### Good Signs
- Loss drops quickly in first 100 iterations
- Validation tracks training closely
- Gradient norm stays < 0.5
- Perplexity < 10 by iteration 200

### Warning Signs  
- Validation diverges after iteration 400 (overfitting)
- Loss spikes randomly (reduce LR)
- Gradient norm > 2.0 (increase clipping)

## Why This Works Better

1. **Small Model Benefits**: 112M params adapt quickly with frequent updates
2. **Quality Data**: 75k diverse examples prevent overfitting
3. **Proper Masking**: Only learning responses = faster convergence
4. **Balanced LR**: 7e-5 is perfect for this update frequency
5. **No Waste**: Every example contributes exactly once

## The Bottom Line

- **772 iterations = 1 complete pass through data**
- **~65 seconds total training time**
- **Best model likely appears at 400-600 iterations**
- **Simple, clean, and efficient**

This approach treats your 112M model appropriately - as a small, efficient learner that benefits from frequent feedback rather than massive batches!