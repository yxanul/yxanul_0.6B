# Learning Rate Optimization with Curriculum

## DeepSpeed's Key Finding
**4x higher learning rates are stable with sequence curriculum** because:
1. Short sequences → simpler loss landscape
2. Many examples → better gradient estimates  
3. Gradual complexity → model adapts progressively

## Current vs Optimized Learning Rates

### Traditional Approach (No Curriculum)
```yaml
learning_rate: 6e-4  # Conservative for 2048 tokens
warmup_steps: 2000   # Long warmup for stability
```

### DeepSpeed-Inspired (With Curriculum)
```yaml
peak_learning_rate: 2.4e-3  # 4x higher!
warmup_strategy: "adaptive"  # Scales with sequence length
```

## Proposed Learning Rate Schedule

### Stage-Aware LR Scaling
```python
def get_stage_lr(base_lr, seq_len, batch_size):
    """
    Scale LR based on:
    1. Sequence length (shorter = higher LR safe)
    2. Batch size (larger = higher LR needed)
    """
    # Microsoft's formula (simplified)
    seq_factor = min(2048 / seq_len, 4.0)  # Up to 4x for short seqs
    batch_factor = math.sqrt(batch_size / 32)  # Linear scaling rule
    
    return base_lr * seq_factor * batch_factor
```

### Curriculum-Aligned Schedule

| Stage | Seq Len | Batch | Base LR | Seq Factor | Batch Factor | Final LR |
|-------|---------|-------|---------|------------|--------------|----------|
| 0-2k  | 64      | 256   | 6e-4    | 4.0x       | 2.83x        | **6.8e-3** |
| 2k-4k | 128     | 128   | 6e-4    | 3.5x       | 2.0x         | **4.2e-3** |
| 4k-8k | 256     | 64    | 6e-4    | 2.5x       | 1.41x        | **2.1e-3** |
| 8k-20k| 512     | 32    | 6e-4    | 1.5x       | 1.0x         | **9e-4**   |
| 20k+  | 1024    | 16    | 6e-4    | 1.0x       | 0.71x        | **4.2e-4** |

## Adaptive Warmup Strategy

### Token-Based Warmup (DeepSpeed Style)
```python
def adaptive_warmup_schedule(current_step, current_seq_len):
    """
    Shorter sequences need less warmup
    """
    # Base warmup tokens (not steps!)
    warmup_tokens = 100_000_000  # 100M tokens
    
    # Current tokens per step
    tokens_per_step = batch_size * current_seq_len
    
    # Warmup steps for this stage
    warmup_steps = warmup_tokens / tokens_per_step
    
    # But cap it reasonably
    return min(warmup_steps, 2000)
```

### Implementation:
```yaml
# Per-stage warmup
curriculum_stages:
  - step: 0
    seq_len: 64
    batch_size: 256
    peak_lr: 6.8e-3  # Aggressive!
    warmup_steps: 500  # Only 500 for short seqs
    
  - step: 2000
    seq_len: 128
    batch_size: 128
    peak_lr: 4.2e-3
    warmup_steps: 300  # Already warmed
    
  - step: 4000
    seq_len: 256
    batch_size: 64
    peak_lr: 2.1e-3
    warmup_steps: 200
```

## Safety Mechanisms

### 1. Gradient Clipping (Adaptive)
```python
def get_grad_clip(seq_len):
    """More aggressive clipping for longer sequences"""
    if seq_len <= 128:
        return 5.0  # Loose clipping for short seqs
    elif seq_len <= 512:
        return 1.0  # Standard
    else:
        return 0.5  # Tight for long seqs
```

### 2. Loss Spike Detection
```python
if loss > 2 * running_avg_loss:
    # Spike detected! Reduce LR temporarily
    current_lr *= 0.5
    print(f"Loss spike! Reducing LR to {current_lr}")
```

### 3. LR Finder First (Optional)
```python
# Run 1000 steps with exponentially increasing LR
# Find the "edge of stability" for each seq length
for seq_len in [64, 128, 256, 512, 1024]:
    max_stable_lr[seq_len] = find_max_lr(seq_len)
```

## Optimized Configuration

```yaml
# stage1_curriculum_extreme_lr.yaml
training:
  # Base LR (will be scaled per stage)
  base_learning_rate: 6e-4
  
  # Aggressive curriculum with LR scaling
  curriculum_stages:
    - step: 0
      seq_len: 64
      batch_size: 256
      lr_multiplier: 11.3  # 6.8e-3 effective
      warmup_steps: 500
      grad_clip: 5.0
      
    - step: 2000
      seq_len: 128
      batch_size: 128
      lr_multiplier: 7.0   # 4.2e-3 effective
      warmup_steps: 300
      grad_clip: 2.0
      
    - step: 4000
      seq_len: 256
      batch_size: 64
      lr_multiplier: 3.5   # 2.1e-3 effective
      warmup_steps: 200
      grad_clip: 1.0
      
    - step: 8000
      seq_len: 512
      batch_size: 32
      lr_multiplier: 1.5   # 9e-4 effective
      warmup_steps: 100
      grad_clip: 0.5
      
    - step: 20000
      seq_len: 1024
      batch_size: 16
      lr_multiplier: 0.7   # 4.2e-4 effective
      warmup_steps: 100
      grad_clip: 0.3
  
  # LR scheduler within each stage
  lr_scheduler_type: "cosine"
  min_lr_ratio: 0.1  # Min is 10% of stage peak
```

## Expected Benefits

### Training Speed
- **10-15% faster convergence** from higher LR
- **Better early learning** with aggressive rates
- **Smoother loss curve** with adaptive warmup

### Quality
- **Sharper minima** from high LR exploration
- **Better generalization** (high LR = implicit regularization)
- **Less overfitting** early on

## Implementation Priority

1. **Stage-based LR multipliers** (easiest)
2. **Adaptive warmup per stage** (important)
3. **Dynamic gradient clipping** (safety)
4. **Loss spike detection** (optional but nice)

## Key Formula

```python
effective_lr = base_lr * sqrt(batch_size/32) * min(2048/seq_len, 4.0)
```

This gives us **11x higher LR** for 64-token sequences with batch=256!

## Warning Signs to Watch

- Loss spikes > 2x average
- Gradient norm explosion
- NaN losses (reduce LR immediately)
- Validation diverging from train

## TL;DR for Your Next Run

1. **Start with LR = 6.8e-3** for 64-token stage (11x traditional!)
2. **Use only 500 warmup steps** initially
3. **Scale LR down as sequences grow**
4. **Monitor gradient norms closely**
5. **Have fallback to conservative LR if unstable**

The key insight: **"Short sequences are training wheels that let you pedal faster!"**