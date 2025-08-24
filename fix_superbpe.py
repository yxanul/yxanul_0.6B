#!/usr/bin/env python
"""
Fix SuperBPE tokenizer loading issue
Run this before training to download and cache the tokenizer
"""

import os
import sys

# Set environment variable if not set
if 'HF_TOKEN' not in os.environ:
    print("Please set HF_TOKEN environment variable first!")
    print("export HF_TOKEN=your_huggingface_token_here")
    sys.exit(1)

print("Attempting to load SuperBPE tokenizer with compatibility fixes...")

try:
    # Try with transformers 4.36.2 which should work
    from transformers import AutoTokenizer
    
    # Try t=80k first (best performance)
    try:
        print("Loading SuperBPE-t80k (37.5% token reduction)...")
        tokenizer = AutoTokenizer.from_pretrained(
            "UW/OLMo2-8B-SuperBPE-t80k",
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
            use_fast=False  # Try slow tokenizer if fast fails
        )
        print(f"✓ SuperBPE-t80k loaded successfully!")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Efficiency: 7.184 chars/token")
        
        # Save locally for offline use
        tokenizer.save_pretrained("./tokenizer_cache/superbpe-t80k")
        print("✓ Tokenizer cached locally in ./tokenizer_cache/superbpe-t80k")
        
    except Exception as e:
        print(f"Could not load t80k: {e}")
        print("Trying t180k fallback...")
        
        # Try t=180k as fallback
        tokenizer = AutoTokenizer.from_pretrained(
            "UW/OLMo2-8B-SuperBPE-t180k",
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
            use_fast=False
        )
        print(f"✓ SuperBPE-t180k loaded as fallback")
        print(f"  Vocab size: {len(tokenizer)}")
        tokenizer.save_pretrained("./tokenizer_cache/superbpe-t180k")
        
except Exception as e:
    print(f"Failed to load SuperBPE: {e}")
    print("\nTroubleshooting steps:")
    print("1. Ensure transformers==4.36.2 is installed")
    print("2. Check your HF_TOKEN is valid")
    print("3. Try: pip install tokenizers==0.15.0")
    sys.exit(1)

print("\n✓ Tokenizer ready! Now run training:")
print("python train_fp8.py --config configs/fineweb_training_ultra_curriculum.yaml")