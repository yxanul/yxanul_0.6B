# TinyStories Training Experiments

This document tracks experimental results for different training configurations on the TinyStories dataset.

## Setup
- **Dataset**: TinyStories (474M train tokens, 4.8M validation tokens)
- **Model**: GPT-2 small variant (113M parameters)
- **Hardware**: NVIDIA RTX 5090 (32GB VRAM)
- **Framework**: PyTorch 2.6 with Flash Attention
- **Batch Configuration**: batch_size=64, gradient_accumulation=16 (effective batch=1024)
- **Sequence Length**: 128 tokens
- **Evaluation**: Every 500 iterations

---

## Experiment 1: BF16 Baseline
**Date**: December 2024  
**Config**: `train_tinystories.py --dtype bfloat16 --max_iters 1000`  
**Purpose**: Establish baseline performance with BF16 mixed precision

### Training Metrics
```
iter 0:    loss 10.9652, time 6506.00ms,   0 tok/s,        lr 0.00e+00
iter 100:  loss 7.5410,  time 112196.47ms, 116,824 tok/s,  lr 1.00e-05
iter 200:  loss 5.8260,  time 112355.13ms, 116,659 tok/s,  lr 2.00e-05
iter 300:  loss 4.5633,  time 112401.84ms, 116,610 tok/s,  lr 3.00e-05
iter 400:  loss 3.8339,  time 112425.90ms, 116,585 tok/s,  lr 4.00e-05

Step 500:  train loss 3.2190, val loss 3.2224
Saved new best model (val loss: 3.2224)

iter 500:  loss 3.1840,  time 118409.63ms, 110,694 tok/s,  lr 5.00e-05
iter 600:  loss 3.0039,  time 112381.19ms, 116,632 tok/s,  lr 6.00e-05
iter 700:  loss 2.7352,  time 112371.28ms, 116,642 tok/s,  lr 7.00e-05
iter 800:  loss 2.5063,  time 112340.96ms, 116,673 tok/s,  lr 8.00e-05
iter 900:  loss 2.2073,  time 112408.63ms, 116,603 tok/s,  lr 9.00e-05

Step 1000: train loss 2.1456, val loss 2.1477
Saved new best model (val loss: 2.1477)

iter 1000: loss 2.0410,  time 118459.17ms, 110,647 tok/s,  lr 1.00e-04
```

### Results Summary
- **Final Loss**: Train 2.1456, Validation 2.1477
- **Perplexity**: ~8.5 (exp(2.14))
- **Average Speed**: 116,000 tokens/sec
- **Total Time**: ~19 minutes for 1000 iterations
- **GPU Power**: 500W sustained
- **GPU Utilization**: 100%
- **Memory Usage**: 23GB / 32GB

### Generation Quality
Model produces coherent children's stories with proper grammar and narrative structure. Minor issues with pronoun consistency and occasional logic jumps, but overall excellent quality for just 1000 iterations.

Sample generation:
```
Prompt: "Once upon a time there was a little"
Output: "Once upon a time there was a little girl named Lily. She loved to play outside in the sunshine. 
One day, she found a beautiful butterfly in her garden. The butterfly had colorful wings and 
danced in the air. Lily wanted to catch it, but her mother told her to let it fly free. 
She learned that some things are more beautiful when they are free."
```

### Key Observations
1. **Extremely fast convergence**: Loss dropped from 10.96 → 2.14 in just 1000 iterations
2. **Perfect train/val alignment**: No overfitting observed (train ≈ val loss)
3. **Consistent throughput**: Maintained 116k tokens/sec throughout training
4. **Stable training**: No loss spikes or instabilities with BF16

---

## Planned Experiments

### Experiment 2: FP8 Mixed Precision
- Use TransformerEngine for FP8 compute
- Target: 2x speedup vs BF16 baseline
- Expected: ~230k tokens/sec

### Experiment 3: FP4/NVF4 (RTX 5090 Specific)
- Leverage RTX 5090's 4-bit capabilities
- Target: 4x speedup vs BF16 baseline
- Expected: ~450k tokens/sec (theoretical)

### Experiment 4: Sliding Window Attention
- Implement local attention with window size 64
- Reduce memory usage and potentially increase speed
- Compare quality vs full attention

### Experiment 5: Different Model Sizes
- Test 50M, 200M, 350M parameter variants
- Find optimal size/speed/quality tradeoff for TinyStories

### Experiment 6: Longer Training
- Extend to 5k iterations to see if quality improves
- Check if loss plateaus or continues declining

### Experiment 7: Different Learning Rates
- Test without warmup (direct to 1e-4)
- Test cosine schedule vs linear
- Test higher peak LR (3e-4)

---

## Conclusions

1. **TinyStories is ideal for rapid experimentation**: Full training runs in under 20 minutes
2. **113M parameters may be oversized**: Dataset is simple enough that smaller models might suffice
3. **BF16 provides stable, fast training**: Excellent baseline for comparisons
4. **RTX 5090 delivers impressive performance**: 116k tokens/sec is 2.8x faster than expected

The fast convergence and low final loss suggest TinyStories has very learnable patterns, making it perfect for testing architectural changes and optimization strategies.