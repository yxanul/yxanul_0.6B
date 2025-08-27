# Simple, Clean Training Implementation

A battle-tested, debuggable implementation without over-engineering.

## Philosophy

- **Simple is better than complex**
- **Explicit is better than implicit**
- **Debuggable is better than clever**
- **Working is better than optimal**

## Features

✅ **Clean code** - Easy to read, modify, and debug  
✅ **Flash Attention** - Automatically uses Flash Attention 2 via PyTorch's SDPA  
✅ **Modern architecture** - RoPE, RMSNorm, SwiGLU, GQA  
✅ **Fast data loading** - Memory-mapped files like nanoGPT  
✅ **Stable training** - BF16, gradient clipping, proper initialization  
✅ **Good monitoring** - Loss tracking, tokens/sec, checkpointing  

## Quick Start

### 1. Prepare Data

```bash
cd experimental
python prepare_data.py
```

This converts your parquet dataset to memory-mapped binary files for fast loading.

### 2. Train Model

```bash
python train.py
```

That's it! No complex configs, no inheritance hierarchies, just training.

## Configuration

All configuration is at the top of `train.py`. Just edit the variables:

```python
# Model
vocab_size = 50257  # GPT-2 vocab

# Data
batch_size = 4
block_size = 2048
gradient_accumulation_steps = 32  # Effective batch = 128

# Training
max_iters = 100000
learning_rate = 3e-4
```

## Model Architecture

- **Parameters**: ~270M
- **Layers**: 24
- **Hidden size**: 1024
- **Attention heads**: 16 (with 4 KV heads for GQA)
- **Context length**: 2048
- **Vocabulary**: 50,257 (GPT-2)

## Performance

Expected performance on RTX 5090:
- **Tokens/sec**: 40,000-60,000
- **Memory usage**: ~8-10 GB
- **Time to convergence**: ~6-12 hours

## Files

- `prepare_data.py` - Convert parquet to memmap
- `model.py` - Clean transformer implementation  
- `train.py` - Simple training loop
- `run.sh` - Helper script to run everything

## Key Differences from Main Codebase

| Feature | Main Codebase | This Implementation |
|---------|--------------|---------------------|
| Config files | 10+ YAML files | Variables in train.py |
| Model files | 5+ files, complex inheritance | 1 file, 400 lines |
| FP8 | Complex TransformerEngine integration | Just BF16 (stable) |
| Data loading | Complex streaming/caching | Simple memmap |
| Curriculum | 10-stage complex scheduling | Fixed batch size |
| Lines of code | ~5000 | ~800 |

## Tips

1. **Start with GPT-2 tokenizer** (50k vocab) - Much easier on memory than 200k
2. **Use BF16** - More stable than FP8, nearly as fast
3. **Fixed batch sizes** - No curriculum complexity
4. **Monitor GPU usage** - Should be >90% utilization
5. **Check tokens/sec** - Should be 40k+ on RTX 5090

## Debugging

If something goes wrong:

1. **Check data**: 
   ```python
   X, Y = get_batch('train')
   print(X.shape, Y.shape)
   ```

2. **Check model**:
   ```python
   logits, loss = model(X, Y)
   print(loss.item())
   ```

3. **Check memory**:
   ```python
   print(torch.cuda.memory_allocated()/1e9)
   ```

4. **Check speed**:
   - Look for "tokens/sec" in output
   - Should be 40,000+

## Why This Works

1. **Memory-mapped data** - OS handles caching, no Python overhead
2. **Flash Attention** - PyTorch's F.scaled_dot_product_attention handles it
3. **Simple optimizer** - AdamW with cosine schedule, no tricks
4. **Proper initialization** - Scaled init for residual connections
5. **No over-engineering** - Just the code that actually runs

## Next Steps

Once this works well, you can:
1. Try larger vocab (use SuperBPE tokenizer)
2. Increase model size (change n_layer, n_embd)
3. Add more monitoring (WandB, etc.)
4. Try torch.compile for extra speed

But get the simple version working first!