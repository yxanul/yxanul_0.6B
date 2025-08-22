# Complete Training Plan with Enhanced Curriculum

## Dataset Overview
- **Size**: 237,758 Wikipedia articles
- **Tokens**: ~920M total
- **Train/Val Split**: 225,870 train / 11,888 validation

## Enhanced 9-Stage Curriculum

### Stage Progression (Now with 768 and 1536!)

| Step Range | Seq Len | Batch | LR (×base) | Chunks/Doc | Steps/Epoch | Examples/Step |
|------------|---------|-------|------------|------------|-------------|---------------|
| 0-2k       | 64      | 256   | 11.3×      | ~60        | 55,725      | 256           |
| 2k-4k      | 128     | 128   | 7.0×       | ~30        | 55,725      | 128           |
| 4k-8k      | 256     | 64    | 3.5×       | ~15        | 55,725      | 64            |
| 8k-12k     | 512     | 32    | 1.5×       | ~7.5       | 55,725      | 32            |
| **12k-18k** | **768** | **21** | **1.0×**  | **~5**     | **53,540**  | **21**        |
| 18k-25k    | 1024    | 16    | 0.7×       | ~3.7       | 54,973      | 16            |
| **25k-35k** | **1536** | **10** | **0.5×**  | **~2.5**   | **59,470**  | **10**        |
| 35k-50k    | 2048    | 8     | 0.4×       | ~1.9       | 56,468      | 8             |
| 50k+       | 2048    | 8     | 0.3×       | ~1.9       | 56,468      | 8             |

### Timeline Breakdown

```
PHASE 1: FOUNDATION (0-12k steps) - "Learn the Basics"
═══════════════════════════════════════════════════════
Steps 0-2k (64 tokens):
  • 512,000 examples (2.3× dataset coverage)
  • 32.8M tokens
  • Learn: Token relationships, common words
  • Time: ~20 minutes

Steps 2k-4k (128 tokens):
  • 256,000 examples (1.1× dataset)
  • 32.8M tokens
  • Learn: Phrases, basic grammar
  • Time: ~20 minutes

Steps 4k-8k (256 tokens):
  • 256,000 examples (1.1× dataset)
  • 65.5M tokens
  • Learn: Sentences, punctuation
  • Time: ~40 minutes

Steps 8k-12k (512 tokens):
  • 128,000 examples (0.6× dataset)
  • 65.5M tokens
  • Learn: Paragraph structure
  • Time: ~40 minutes

Checkpoint: 2 hours, 1.15M examples, PPL should be < 500

PHASE 2: COMPREHENSION (12k-35k steps) - "Understand Context"
═══════════════════════════════════════════════════════════════
Steps 12k-18k (768 tokens):
  • 126,000 examples (0.5× dataset)
  • 96.8M tokens
  • Learn: Multi-paragraph flow
  • Time: ~1 hour

Steps 18k-25k (1024 tokens):
  • 112,000 examples (0.5× dataset)
  • 114.7M tokens
  • Learn: Section relationships
  • Time: ~1.2 hours

Steps 25k-35k (1536 tokens):
  • 100,000 examples (0.4× dataset)
  • 153.6M tokens
  • Learn: Article structure
  • Time: ~1.6 hours

Checkpoint: 6 hours total, 1.6M examples, PPL should be < 50

PHASE 3: MASTERY (35k-100k steps) - "Deep Understanding"
═══════════════════════════════════════════════════════════
Steps 35k-50k (2048 tokens):
  • 120,000 examples (0.5× dataset)
  • 245.8M tokens
  • Learn: Full document coherence
  • Time: ~2.5 hours

Steps 50k-100k (2048 tokens):
  • 400,000 examples (1.7× dataset)
  • 819.2M tokens
  • Learn: Deep knowledge integration
  • Time: ~8 hours

Final: ~17 hours total, 2.3M examples, PPL target < 25
```

## Validation: How It Actually Works

### What Gets Validated?

**11,888 held-out Wikipedia articles** that the model has NEVER seen during training.

### Validation Process (Every 2000 steps):

```python
# 1. Switch model to eval mode (disables dropout)
model.eval()

# 2. Process validation articles
for batch in validation_dataloader:
    # Same chunking as training (seq_len matches current stage)
    input_ids = batch["input_ids"]  # [16, current_seq_len]
    
    # 3. Forward pass (no backprop)
    with torch.no_grad():
        outputs = model(input_ids)
        loss = outputs.loss
        logits = outputs.logits
    
    # 4. Calculate metrics
    perplexity = torch.exp(loss)
    
    # 5. Accuracy: Does model predict next token correctly?
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
```

### Validation Metrics Explained:

1. **Perplexity (Main Metric)**
   ```
   PPL = exp(average_loss)
   
   Good: < 50
   Great: < 30
   Excellent: < 20
   ```
   Measures: "How surprised is the model by unseen text?"

2. **Loss**
   ```
   Cross-entropy loss on next-token prediction
   Lower = better
   ```

3. **Accuracy**
   ```
   % of next tokens predicted correctly
   
   Random: ~0.002% (1/50k vocabulary)
   Good: > 10%
   Great: > 20%
   ```

### Why Validation Matters:

```
Training Loss vs Validation Loss:

Good (No overfitting):
Train PPL: 25 ←─ Small gap
Val PPL:   30 ←─ Good generalization

Bad (Overfitting):
Train PPL: 15 ←─ Large gap!
Val PPL:   60 ←─ Memorizing, not learning
```

### Validation with Curriculum:

The validation uses the SAME sequence length as current training stage:

```python
# At step 5000 (seq_len = 256):
validation_chunks = create_chunks(val_articles, seq_len=256)

# At step 25000 (seq_len = 1536):
validation_chunks = create_chunks(val_articles, seq_len=1536)
```

This means:
- **Early validation is easier** (short sequences)
- **Late validation is harder** (long sequences)
- **PPL may temporarily spike** when sequence length increases

### Expected Validation Timeline:

| Step  | Train Seq | Val PPL | Why |
|-------|-----------|---------|-----|
| 2k    | 64        | ~5000   | Just starting to learn |
| 5k    | 256       | ~1000   | Basic patterns learned |
| 10k   | 512       | ~300    | Grammar understood |
| 15k   | 768       | ~150    | Context emerging |
| 20k   | 1024      | ~80     | Good comprehension |
| 30k   | 1536      | ~50     | Strong model |
| 50k   | 2048      | ~35     | Excellent |
| 100k  | 2048      | ~25     | Near optimal |

### Early Stopping Logic:

```python
if val_perplexity > best_val_perplexity:
    patience_counter += 1
    if patience_counter > 10:  # No improvement for 10 evals
        print("Early stopping triggered!")
        break
else:
    best_val_perplexity = val_perplexity
    save_checkpoint("best_model.pt")
```

## Updated Configuration with 768 & 1536

```yaml
# stage1_curriculum_final.yaml
curriculum_stages:
  - {step: 0,     seq_len: 64,   batch_size: 256, lr_scale: 11.3}
  - {step: 2000,  seq_len: 128,  batch_size: 128, lr_scale: 7.0}
  - {step: 4000,  seq_len: 256,  batch_size: 64,  lr_scale: 3.5}
  - {step: 8000,  seq_len: 512,  batch_size: 32,  lr_scale: 1.5}
  - {step: 12000, seq_len: 768,  batch_size: 21,  lr_scale: 1.0}  # NEW!
  - {step: 18000, seq_len: 1024, batch_size: 16,  lr_scale: 0.7}
  - {step: 25000, seq_len: 1536, batch_size: 10,  lr_scale: 0.5}  # NEW!
  - {step: 35000, seq_len: 2048, batch_size: 8,   lr_scale: 0.4}
  - {step: 50000, seq_len: 2048, batch_size: 8,   lr_scale: 0.3}
```

## Key Insights

1. **Validation frequency**: Every 2000 steps is perfect
   - Not too often (expensive)
   - Not too rare (miss problems)

2. **Validation tells you**:
   - Is training working? (PPL dropping)
   - Are we overfitting? (Train/val gap)
   - When to stop? (Val plateaus)

3. **With curriculum, expect**:
   - Validation PPL drops fast initially
   - Small spikes when seq_len increases
   - Smooth improvement after adjustments

## Summary

Your optimized plan processes:
- **2.3M examples** in 100k steps
- **1.8B tokens** total
- **17 hours** on RTX 4090
- **9 curriculum stages** for smooth learning

This is **7.7× more examples** than traditional training for the same compute!