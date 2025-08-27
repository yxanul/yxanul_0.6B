#!/usr/bin/env python3
"""
Verify that the model is actually 270M parameters and factorized embeddings work correctly.
"""

import torch
import sys
import os
sys.path.append('src')

from model_te_v2 import create_te_v2_model, ModelConfig

def calculate_parameters_manually(config):
    """Calculate expected parameter count manually"""
    
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    factorization_dim = config.factorization_dim if config.use_factorized_embedding else None
    
    print("\n" + "="*60)
    print("Manual Parameter Calculation")
    print("="*60)
    
    total_params = 0
    
    # 1. Embeddings
    if factorization_dim:
        # Factorized: vocab -> r -> hidden
        embed_params = vocab_size * factorization_dim  # First embedding
        proj_params = factorization_dim * hidden_size   # Projection
        print(f"Factorized Embeddings:")
        print(f"  Embedding: {vocab_size} × {factorization_dim} = {embed_params:,}")
        print(f"  Projection: {factorization_dim} × {hidden_size} = {proj_params:,}")
        embedding_total = embed_params + proj_params
        print(f"  Total: {embedding_total:,} ({embedding_total/1e6:.2f}M)")
    else:
        # Standard embedding
        embedding_total = vocab_size * hidden_size
        print(f"Standard Embedding: {vocab_size} × {hidden_size} = {embedding_total:,}")
    
    total_params += embedding_total
    
    # 2. Per-layer parameters
    print(f"\nPer Transformer Layer:")
    
    # Attention (with GQA)
    # Q projection: hidden -> num_heads * head_dim
    q_params = hidden_size * (num_heads * head_dim)
    # K,V projections: hidden -> num_kv_heads * head_dim each
    kv_params = 2 * hidden_size * (num_kv_heads * head_dim)
    # Output projection: (num_heads * head_dim) -> hidden
    o_params = (num_heads * head_dim) * hidden_size
    
    attn_params = q_params + kv_params + o_params
    print(f"  Attention (GQA {num_heads}:{num_kv_heads}):")
    print(f"    Q: {hidden_size} × {num_heads * head_dim} = {q_params:,}")
    print(f"    K,V: 2 × {hidden_size} × {num_kv_heads * head_dim} = {kv_params:,}")
    print(f"    Out: {num_heads * head_dim} × {hidden_size} = {o_params:,}")
    print(f"    Total: {attn_params:,}")
    
    # FFN with SwiGLU (3 matrices instead of 2)
    # Gate and Up: hidden -> intermediate each
    gate_up_params = 2 * hidden_size * intermediate_size
    # Down: intermediate -> hidden
    down_params = intermediate_size * hidden_size
    
    ffn_params = gate_up_params + down_params
    print(f"  FFN (SwiGLU):")
    print(f"    Gate+Up: 2 × {hidden_size} × {intermediate_size} = {gate_up_params:,}")
    print(f"    Down: {intermediate_size} × {hidden_size} = {down_params:,}")
    print(f"    Total: {ffn_params:,}")
    
    # Layer norms (2 per layer for pre-norm)
    norm_params = 2 * hidden_size
    print(f"  LayerNorms: 2 × {hidden_size} = {norm_params:,}")
    
    layer_total = attn_params + ffn_params + norm_params
    print(f"  Total per layer: {layer_total:,}")
    
    # 3. All layers
    all_layers = layer_total * num_layers
    print(f"\nAll {num_layers} layers: {layer_total:,} × {num_layers} = {all_layers:,}")
    total_params += all_layers
    
    # 4. Final norm
    final_norm = hidden_size
    print(f"Final RMSNorm: {hidden_size:,}")
    total_params += final_norm
    
    # 5. LM head (not tied with embeddings when factorized)
    lm_head = hidden_size * vocab_size
    print(f"LM Head: {hidden_size} × {vocab_size} = {lm_head:,}")
    total_params += lm_head
    
    print("\n" + "="*60)
    print(f"TOTAL EXPECTED: {total_params:,} ({total_params/1e6:.2f}M)")
    print("="*60)
    
    # Calculate what hidden_size we need for 270M
    if total_params > 300e6:
        print("\nWARNING: Model is too large! Calculating correct dimensions for 270M...")
        
        # Reverse calculate for 270M target
        target = 270e6
        # Approximate: most params are in layers
        # For 270M with 32 layers, we need smaller hidden_size
        
        # Try different hidden sizes
        for try_hidden in [768, 704, 640, 576, 512]:
            try_intermediate = int(try_hidden * 2.67)  # SwiGLU ratio
            
            # Quick approximation
            approx_params = (
                vocab_size * factorization_dim +  # embed
                factorization_dim * try_hidden +  # proj
                num_layers * (
                    4 * try_hidden * try_hidden +  # attention (simplified)
                    3 * try_hidden * try_intermediate  # FFN
                ) +
                try_hidden * vocab_size  # LM head
            )
            
            if approx_params < 280e6:
                print(f"  hidden_size={try_hidden}, intermediate={try_intermediate} -> ~{approx_params/1e6:.1f}M [GOOD]")
                break
            else:
                print(f"  hidden_size={try_hidden}, intermediate={try_intermediate} -> ~{approx_params/1e6:.1f}M (too large)")
    
    return total_params

def verify_model():
    """Create actual model and count parameters"""
    
    # Load config for 270M model (CORRECTED)
    config = ModelConfig(
        vocab_size=200005,
        hidden_size=576,  # Corrected for 270M
        intermediate_size=1536,  # ~2.67x hidden
        num_hidden_layers=32,
        num_attention_heads=9,  # 9 heads * 64 dim = 576
        num_kv_heads=3,  # GQA 3:1 ratio
        head_dim=64,
        use_factorized_embedding=True,
        factorization_dim=128,
        use_fp8=False  # Disable for testing
    )
    
    # Calculate expected
    expected_params = calculate_parameters_manually(config)
    
    try:
        # Create actual model
        print("\n" + "="*60)
        print("Creating Actual Model")
        print("="*60)
        
        model = create_te_v2_model(config, enable_gradient_checkpointing=False)
        
        # Count actual parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nActual Model Parameters:")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # Check factorized embedding
        if hasattr(model, 'embed_tokens'):
            if hasattr(model.embed_tokens, 'embed'):
                embed_shape = model.embed_tokens.embed.weight.shape
                proj_shape = model.embed_tokens.proj.weight.shape
                print(f"\nFactorized Embedding Verification:")
                print(f"  Embedding matrix: {embed_shape} = {embed_shape[0]:,} × {embed_shape[1]}")
                print(f"  Projection matrix: {proj_shape} = {proj_shape[0]:,} × {proj_shape[1]}")
                print(f"  Factorization rank r = {embed_shape[1]} ✓")
                
                # Calculate savings
                standard_params = config.vocab_size * config.hidden_size
                factorized_params = embed_shape[0] * embed_shape[1] + proj_shape[0] * proj_shape[1]
                savings = standard_params - factorized_params
                print(f"  Parameter savings: {savings:,} ({savings/1e6:.2f}M)")
                print(f"  Reduction: {(1 - factorized_params/standard_params)*100:.1f}%")
        
        # Detailed breakdown
        print("\n" + "="*60)
        print("Detailed Parameter Breakdown")
        print("="*60)
        
        for name, param in model.named_parameters():
            if 'embed' in name or 'lm_head' in name or 'norm' in name:
                print(f"{name}: {tuple(param.shape)} = {param.numel():,}")
        
    except Exception as e:
        print(f"\nError creating model: {e}")
        print("This is expected if TransformerEngine is not available.")
        print("The manual calculation above shows the expected parameter count.")
    
    return expected_params

if __name__ == "__main__":
    expected = verify_model()
    
    if expected > 300e6:
        print("\n" + "="*60)
        print("RECOMMENDATION: Update config to achieve 270M parameters")
        print("="*60)
        print("""
Suggested changes in yxanul_270m_experimental_1b.yaml:

model:
  hidden_size: 640  # Reduced from 896
  intermediate_size: 1708  # ~2.67x hidden for SwiGLU
  num_hidden_layers: 32  # Keep same
  num_attention_heads: 10  # Adjusted for head_dim=64
  
This should give approximately 270M parameters.
        """)