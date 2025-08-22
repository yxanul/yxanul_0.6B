# Complete Multi-Stage Training Plan

## Overview: 4-Stage Progressive Training
Total: ~100B tokens across specialized datasets

```
Stage 1: FOUNDATION (Wikipedia) → Language understanding
Stage 2: REASONING (MoT/DeepSeek) → How to think
Stage 3: MATHEMATICS (GSM8K) → Numerical reasoning  
Stage 4: CODING (Optional) → Technical skills
```

---

## STAGE 1: Wikipedia Foundation ✅ (Planned)
**Dataset**: Yxanul/wikipedia-2k-high-quality (0.92B tokens)
**Duration**: 15 epochs = 14B tokens
**Config**: stage1_curriculum_optimized.yaml → stage1_full_sequence.yaml

### Execution:
```
Epoch 1: Curriculum (64→2048) - 6-8 hours
Epochs 2-5: Full sequences (2048) - 60 hours
Result: PPL ~20, strong language foundation
```

**Checkpoint**: `model_wikipedia_complete.pt`

---

## STAGE 2: Reasoning Development 🧠
**Dataset**: open-r1/Mixture-of-Thoughts (1.5B tokens)
**Duration**: 20 epochs = 30B tokens
**Purpose**: Learn chain-of-thought reasoning from DeepSeek-R1 traces

### Why This Dataset is Special:
```
Contains actual reasoning traces like:
"Let me think about this step by step...
First, I need to identify...
Wait, that's not right. Let me reconsider...
Actually, the correct approach is..."
```

### Training Strategy:
```yaml
# stage2_reasoning.yaml
stage:
  name: "reasoning_development"
  description: "Learn HOW to think, not just facts"

data:
  dataset_name: "open-r1/Mixture-of-Thoughts"
  max_sequence_length: 4096  # Reasoning chains are long!
  
training:
  # Start from Wikipedia checkpoint
  resume_from_checkpoint: "model_wikipedia_complete.pt"
  
  # Gentle curriculum (model already knows basics)
  use_curriculum: true
  curriculum_stages:
    - {step: 0,     seq_len: 1024, batch_size: 16, lr_scale: 1.0}
    - {step: 5000,  seq_len: 2048, batch_size: 8,  lr_scale: 0.8}
    - {step: 10000, seq_len: 3072, batch_size: 5,  lr_scale: 0.6}
    - {step: 20000, seq_len: 4096, batch_size: 4,  lr_scale: 0.5}
  
  # Lower learning rate (fine-tuning)
  learning_rate: 1e-4  # 50% of Wikipedia peak
  
  # Longer sequences need different settings
  gradient_accumulation_steps: 8  # Effective batch = 32
  gradient_checkpointing: true  # Save memory
  
  per_device_train_batch_size: 4
  num_epochs: 20
  
expected_results:
  # Reasoning capabilities
  - "Multi-step problem solving"
  - "Self-correction during generation"
  - "Explicit thinking patterns"
  perplexity: "< 15 on reasoning tasks"
```

**Time**: ~100 hours (20 epochs × 5 hours)
**Checkpoint**: `model_reasoning_complete.pt`

---

## STAGE 3: Mathematical Thinking 🔢
**Dataset**: GSM8K + Mathematical collections (0.5B tokens)
**Duration**: 30 epochs = 15B tokens (smaller dataset, more epochs)
**Purpose**: Numerical reasoning and calculation

### Special Considerations:
```python
# Math needs different tokenization
# "342 + 567 = 909" should preserve number structure
# May need custom data processing
```

### Training Strategy:
```yaml
# stage3_mathematics.yaml
stage:
  name: "mathematical_reasoning"
  
data:
  datasets:  # Multiple math datasets
    - "gsm8k"
    - "competition_math" 
    - "khan_academy_math"
  max_sequence_length: 2048
  
training:
  resume_from_checkpoint: "model_reasoning_complete.pt"
  
  # No curriculum needed (model is sophisticated)
  use_curriculum: false
  per_device_train_batch_size: 16
  
  # Very low LR to preserve reasoning
  learning_rate: 5e-5  # 25% of Wikipedia
  weight_decay: 0.01  # Less regularization
  
  # Math-specific settings
  num_epochs: 30  # Many passes for small dataset
  eval_steps: 500  # Frequent evaluation
  
  # Special tokens for math
  additional_special_tokens:
    - "<calculation>"
    - "</calculation>"
    - "<step>"
    - "</step>"
    
expected_results:
  - "GSM8K accuracy > 40%"
  - "Basic arithmetic correct > 90%"
  - "Multi-step word problems"
```

**Time**: ~40 hours
**Checkpoint**: `model_math_complete.pt`

---

## STAGE 4: Code Understanding 💻 (Optional)
**Dataset**: bigcode/starcoderdata or similar (5B tokens subset)
**Duration**: 10 epochs = 50B tokens
**Purpose**: Programming and technical knowledge

### Training Strategy:
```yaml
# stage4_code.yaml
stage:
  name: "code_understanding"
  
data:
  dataset_name: "bigcode/starcoderdata"
  languages: ["python", "javascript", "markdown"]
  max_sequence_length: 2048
  
training:
  resume_from_checkpoint: "model_math_complete.pt"
  
  # Code has different structure
  use_curriculum: false
  per_device_train_batch_size: 8
  
  # Preserve all previous knowledge
  learning_rate: 2e-5  # Very low
  weight_decay: 0.001  # Minimal
  
  # Code-specific
  num_epochs: 10
  ignore_data_skip: false  # Don't skip partial code
```

**Time**: ~80 hours
**Checkpoint**: `model_final_complete.pt`

---

## Stage Transition Strategy

### After Each Stage:
1. **Evaluate on benchmarks**:
   ```python
   - HellaSwag (reasoning)
   - GSM8K (math)
   - HumanEval (code)
   - MMLU (knowledge)
   ```

2. **Check for catastrophic forgetting**:
   ```python
   # Test on Wikipedia validation
   if wiki_perplexity > original * 1.5:
       print("Warning: Forgetting detected!")
       # Reduce learning rate
       # Add Wikipedia replay buffer
   ```

3. **Decision point**:
   - Continue to next stage?
   - Need more epochs?
   - Adjust hyperparameters?

---

## Learning Rate Decay Across Stages

| Stage | Base LR | Rationale |
|-------|---------|-----------|
| Wikipedia | 6e-4 | Aggressive, learning from scratch |
| Reasoning | 1e-4 | Moderate, building on foundation |
| Mathematics | 5e-5 | Gentle, preserving reasoning |
| Code | 2e-5 | Minimal, fine-tuning only |

---

## Total Training Timeline

```
Stage 1 (Wikipedia):    75 hours  → Strong foundation
Stage 2 (Reasoning):   100 hours  → How to think
Stage 3 (Math):         40 hours  → Numerical skills
Stage 4 (Code):         80 hours  → Technical knowledge
                      -----------
Total:                 295 hours (~12 days)
```

---

## Optimization Tips

### 1. Interleaved Training (Advanced):
Instead of sequential stages, interleave datasets:
```python
batch_sources = {
    "wikipedia": 0.4,
    "reasoning": 0.3,
    "math": 0.2,
    "code": 0.1
}
```

### 2. Replay Buffers:
Keep 10% of batches from previous stages:
```python
batch = 0.9 * current_stage + 0.1 * replay_buffer
```

### 3. Dynamic Evaluation:
```python
if stage == "reasoning":
    eval_on = ["reasoning_bench", "wikipedia_val"]
elif stage == "math":
    eval_on = ["gsm8k", "reasoning_bench", "wikipedia_val"]
```

---

## Key Insights

1. **Wikipedia First**: Essential foundation, don't skip
2. **Reasoning Before Math**: Abstract thinking before calculation
3. **Decreasing LR**: Preserve previous knowledge
4. **Longer Sequences for Reasoning**: Chains of thought need space
5. **More Epochs for Small Datasets**: GSM8K needs 30 epochs

---

## Recommended Execution Order

### Fast Track (150 hours):
1. Wikipedia: 5 epochs with curriculum
2. Reasoning: 10 epochs
3. Math: 20 epochs
→ Good general model

### Full Training (295 hours):
1. All stages, all epochs
→ SOTA-competitive model

### Experimental (100 hours):
1. Wikipedia: 3 epochs with curriculum
2. Interleaved: Wiki+Reasoning+Math together
→ Potentially better, needs testing