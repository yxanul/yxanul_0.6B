# Optimized SFT Training Configuration

## Dataset Improvements
- **Alpaca**: ~52k instruction-following examples
- **GSM8K**: ~8k math word problems  
- **Dolly-15k**: ~15k diverse tasks (NEW)
  - Summarization
  - Creative writing
  - Information extraction
  - Open/closed QA
  - Classification

**Total**: ~75k examples with high diversity

## Hyperparameter Optimizations

### Learning Rate: 7e-5
**Previous**: 3e-4 (too aggressive) → 1e-4 (still high) → **7e-5 (balanced)**

**Why 7e-5?**
- Preserves pretrained knowledge better
- Reduces catastrophic forgetting
- More stable with 75k diverse examples
- Sweet spot between adaptation and preservation

### Weight Decay: 0.05 with Parameter Groups
**Previous**: 0.1 on all parameters
**Now**: 0.05 on weights only, 0.0 on norms/bias/embeddings

**Excluded from decay**:
- LayerNorms (RMSNorm) - scale sensitive
- Biases (if any) - typically excluded
- Embeddings (wte) - need flexibility
- Position embeddings (rope_cache) - fixed patterns

**With decay**:
- Linear layers (q,k,v,o projections)
- FFN weights (gate, up, down)
- Prevents overfitting on small dataset

### Other Optimizations
- **Warmup**: 10 steps (was 5) - more stable start
- **Min LR**: 1e-5 (was 5e-5) - matches reduced scale
- **Batch**: 16 with 16 accumulation = 256 effective
- **Gradient Clipping**: 1.0 - prevents instability

## Training Command

```bash
# Prepare enhanced dataset
python prepare_clean_sft_data.py \
    --alpaca alpaca_data_cleaned.json \
    --gsm8k train-00000-of-00001.parquet \
    --dolly databricks-dolly-15k.jsonl \
    --output_dir data_sft_enhanced \
    --max_examples 80000  # Optional limit

# Train with optimized settings
python train_sft.py \
    --data_dir data_sft_enhanced \
    --base_model ../train/best_model_fp8_optimized.pt \
    --learning_rate 7e-5 \
    --max_iters 100 \
    --eval_interval 10
```

## Expected Results

With these optimizations:
1. **Lower final loss**: Better convergence with balanced LR
2. **Smoother training**: Proper weight decay prevents spikes
3. **Better generalization**: Dolly diversity improves robustness
4. **Faster convergence**: May need only 50-100 iterations

## Monitoring Tips

Watch for:
- Loss should decrease smoothly (no spikes)
- Perplexity should stay < 10
- Gradient norm should be stable
- Val loss shouldn't diverge from train loss

## Why This Configuration Works

1. **7e-5 LR**: Goldilocks zone for 112M models
2. **Parameter groups**: Respects different parameter sensitivities
3. **Dolly diversity**: Prevents overfitting to single task type
4. **0.05 WD**: Enough regularization without over-constraining

This configuration balances:
- **Adaptation** (learning new tasks)
- **Preservation** (keeping pretrained knowledge)
- **Stability** (smooth optimization)
- **Efficiency** (fast convergence)