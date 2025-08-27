#!/usr/bin/env python3
"""Calculate correct intermediate_size for 270M model with SwiGLU"""

def calculate_params(hidden_size=896, intermediate_size=2400, num_layers=32, vocab_size=200005):
    """Calculate total parameters for the model"""
    
    # Embeddings (factorized)
    factorization_dim = 128
    embed_params = vocab_size * factorization_dim + factorization_dim * hidden_size
    lm_head_params = hidden_size * vocab_size
    
    # Per layer
    # Attention with GQA (14 heads, 2 KV heads)
    head_dim = 64
    num_heads = 14
    num_kv_heads = 2
    
    q_params = hidden_size * hidden_size
    kv_params = 2 * hidden_size * (num_kv_heads * head_dim)
    o_params = hidden_size * hidden_size
    attn_params = q_params + kv_params + o_params
    
    # SwiGLU FFN (3 matrices!)
    ffn_params = 3 * hidden_size * intermediate_size
    
    # LayerNorm (2 per layer)
    ln_params = 2 * hidden_size
    
    # Total per layer
    layer_params = attn_params + ffn_params + ln_params
    
    # Total
    total = embed_params + lm_head_params + (layer_params * num_layers) + hidden_size  # final norm
    
    print(f"Configuration: hidden={hidden_size}, intermediate={intermediate_size}, layers={num_layers}")
    print(f"  Embeddings: {embed_params/1e6:.1f}M")
    print(f"  LM Head: {lm_head_params/1e6:.1f}M") 
    print(f"  Per Layer: {layer_params/1e6:.2f}M")
    print(f"    - Attention: {attn_params/1e6:.2f}M")
    print(f"    - FFN (SwiGLU): {ffn_params/1e6:.2f}M")
    print(f"    - LayerNorm: {ln_params/1e3:.1f}K")
    print(f"  All Layers: {(layer_params * num_layers)/1e6:.1f}M")
    print(f"  TOTAL: {total/1e6:.1f}M\n")
    
    return total

# Current config (too big!)
current = calculate_params(896, 2400, 32)

# Find correct intermediate_size for 270M
target = 270e6
for inter_size in range(2400, 1000, -50):
    total = calculate_params(896, inter_size, 32)
    if total <= target:
        print(f"✓ For 270M model, use intermediate_size={inter_size}")
        calculate_params(896, inter_size, 32)
        break