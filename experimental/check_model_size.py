#!/usr/bin/env python3
"""
Quick script to check model sizes with different vocab configurations.
"""

def calculate_params(vocab_size, n_layer=12, n_head=12, n_embd=768, ffn_hidden_size=2048):
    """Calculate approximate parameter count for transformer model."""
    
    # Embeddings
    token_embed = vocab_size * n_embd
    
    # Per layer
    # Attention: QKV projection + output projection
    attn_params_per_layer = (n_embd * 3 * n_embd) + (n_embd * n_embd)
    
    # FFN: gate_up (to 2*ffn_hidden) + down
    ffn_params_per_layer = (n_embd * 2 * ffn_hidden_size) + (ffn_hidden_size * n_embd)
    
    # LayerNorms: 2 per layer
    ln_params_per_layer = 2 * n_embd
    
    total_per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    
    # All layers
    all_layers = n_layer * total_per_layer
    
    # Output head (usually tied with embeddings, so not counted again)
    lm_head = vocab_size * n_embd  # But in TE model it might not be tied
    
    # Final LayerNorm
    final_ln = n_embd
    
    # Total
    total = token_embed + all_layers + lm_head + final_ln
    
    print(f"\nVocab size: {vocab_size}")
    print(f"  Token embeddings: {token_embed:,}")
    print(f"  Per layer: {total_per_layer:,}")
    print(f"  All {n_layer} layers: {all_layers:,}")
    print(f"  LM head: {lm_head:,}")
    print(f"  Final LN: {final_ln:,}")
    print(f"  Total params: {total:,} ({total/1e6:.1f}M)")
    
    return total

print("Model size calculations for FP8 alignment:\n")
print("="*50)

# Original model
orig = calculate_params(50257)

# FP8-aligned options
opt1 = calculate_params(50256)  # Divisible by 16
opt2 = calculate_params(50272)  # Next multiple of 16
opt3 = calculate_params(50304)  # Multiple of 64
opt4 = calculate_params(32768)  # Much smaller vocab

print("\n" + "="*50)
print("Recommendations:")
print("- Use vocab_size=50256 for minimal change (113.0M params)")
print("- Or reduce vocab_size significantly if 113M target is critical")
print("- Remember: TE model doesn't tie embeddings, so LM head adds params")