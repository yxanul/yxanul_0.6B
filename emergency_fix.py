#!/usr/bin/env python3
"""
EMERGENCY FIX for diverged training at step 28000

Problem diagnosed:
1. Learning rate continuously increasing (now at 5.2e-04, should be ~2e-04 max)
2. Scheduler thinks it's still in warmup phase
3. num_training_steps not properly set, causing infinite warmup

Immediate fixes needed:
1. Restart from checkpoint at step 10000 (before divergence)
2. Fix learning rate schedule
3. Add gradient clipping
4. Reduce max learning rate
"""

import os
import sys

print("="*60)
print("🚨 EMERGENCY TRAINING FIX")
print("="*60)

print("\n📊 DIAGNOSIS:")
print("  • Training diverged around step 12000")
print("  • LR climbed to 5.2e-04 (2.5x too high)")
print("  • Loss stuck at ~9 (random prediction)")
print("  • Model stopped learning completely")

print("\n🔧 ROOT CAUSE:")
print("  The lr_lambda function has no proper num_training_steps set,")
print("  causing infinite warmup. LR keeps increasing forever!")

print("\n✅ SOLUTION:")
print("  Restart from last good checkpoint with fixed config")

print("\n" + "="*60)
print("EXECUTE THIS COMMAND:")
print("="*60)

# Calculate proper training steps for 1 epoch on 1B tokens
tokens_per_batch = 2048  # sequence length
batch_size = 1
gradient_accumulation = 32
tokens_per_optimizer_step = tokens_per_batch * batch_size * gradient_accumulation
total_tokens = 1_000_000_000  # 1B tokens
total_steps = total_tokens // tokens_per_optimizer_step

print(f"""
python train_te_v2.py \\
    --batch-size 2 \\
    --gradient-accumulation 16 \\
    --learning-rate 2e-4 \\
    --warmup-steps 500 \\
    --max-steps {total_steps} \\
    --max-grad-norm 0.5 \\
    --eval-steps 1000 \\
    --save-steps 5000 \\
    --resume-from-checkpoint checkpoints_te_v2/checkpoint_step0010000_epoch000.pt \\
    --config configs/yxanul_270m_fixed_2048.yaml
""")

print("\n📝 KEY CHANGES:")
print("  1. Resume from step 10000 (before divergence)")
print("  2. Fixed learning rate: 2e-4 (not 6e-4)")
print("  3. Shorter warmup: 500 steps (not 1000)")
print("  4. Gradient clipping: 0.5 (more aggressive)")
print("  5. Batch size 2 (better GPU utilization)")
print("  6. Total steps explicitly set for proper decay")

print("\n⚠️  IMPORTANT:")
print("  • This will overwrite the diverged checkpoints")
print("  • Training will restart from step 10000")
print("  • Expected to reach good loss (~4) by step 20000")

print("\n🎯 EXPECTED BEHAVIOR:")
print("  • LR will warm to 2e-4 over 500 steps")
print("  • Then cosine decay to 2e-5 by end")
print("  • Loss should drop below 6 within 2000 steps")
print("  • Validation PPL should reach <1000 by step 30000")

print("\n" + "="*60)

# Also create a fixed trainer config
fixed_config = """
# FIXED trainer kwargs for emergency recovery
trainer_kwargs = {
    'warmup_steps': 500,
    'num_training_steps': 15625,  # Calculated for 1B tokens
    'learning_rate': 2e-4,
    'max_grad_norm': 0.5,
    'batch_size': 2,
    'gradient_accumulation_steps': 16
}
"""

print("\nSaving fixed config to 'emergency_trainer_config.py'...")
with open("emergency_trainer_config.py", "w") as f:
    f.write(fixed_config)

print("Done! Copy and paste the command above to restart training.")
print("\n⏱️  Estimated time to recovery: ~1 hour")
print("💡  Monitor for stable loss decrease in first 1000 steps")