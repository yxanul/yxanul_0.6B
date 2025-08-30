#!/usr/bin/env python3
"""
Analyze the parameter distribution of the trained model.
"""
import torch
from model import ModelConfig, SimpleGPT

def analyze_model_parameters():
    """Detailed parameter breakdown of the model."""
    
    # Create model with exact training configuration
    config = ModelConfig(
        vocab_size=49152,  # SmolLM vocabulary
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=3,  # GQA with 4x compression
        block_size=2048,
        dropout=0.0,
        use_factorized_embedding=False
    )
    
    model = SimpleGPT(config)
    
    print("="*70)
    print("MODEL PARAMETER ANALYSIS")
    print("="*70)
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Embedding parameters
    embedding_params = model.transformer.wte.weight.numel()
    # No position embeddings - model uses RoPE (Rotary Position Embeddings)
    total_embedding_params = embedding_params
    
    print(f"\n{'='*70}")
    print("EMBEDDING LAYERS")
    print(f"{'='*70}")
    print(f"Token Embeddings (wte): {embedding_params:,} ({embedding_params/1e6:.1f}M)")
    print(f"  Shape: {list(model.transformer.wte.weight.shape)} (vocab_size × n_embd)")
    print(f"Position Encoding: RoPE (Rotary Position Embeddings - no parameters)")
    print(f"Total Embeddings: {total_embedding_params:,} ({total_embedding_params/1e6:.1f}M)")
    print(f"Embedding % of model: {total_embedding_params/total_params*100:.1f}%")
    
    # Output head parameters
    lm_head_params = model.lm_head.weight.numel()
    print(f"\n{'='*70}")
    print("OUTPUT HEAD")
    print(f"{'='*70}")
    print(f"LM Head: {lm_head_params:,} ({lm_head_params/1e6:.1f}M)")
    print(f"  Shape: {list(model.lm_head.weight.shape)} (vocab_size × n_embd)")
    print(f"  Note: Weight tied with token embeddings (shared parameters)")
    
    # Transformer core parameters (excluding embeddings and head)
    transformer_params = total_params - total_embedding_params
    print(f"\n{'='*70}")
    print("TRANSFORMER CORE (excluding embeddings)")
    print(f"{'='*70}")
    print(f"Core Transformer: {transformer_params:,} ({transformer_params/1e6:.1f}M)")
    print(f"Core % of model: {transformer_params/total_params*100:.1f}%")
    
    # Per-layer breakdown
    print(f"\n{'='*70}")
    print("PER-LAYER BREAKDOWN")
    print(f"{'='*70}")
    
    # Count parameters in one transformer block
    block_params = 0
    block = model.transformer.h[0]
    
    # Attention parameters
    attn_params = sum(p.numel() for p in block.attn.parameters())
    q_params = block.attn.q_proj.weight.numel()
    k_params = block.attn.k_proj.weight.numel()
    v_params = block.attn.v_proj.weight.numel()
    o_params = block.attn.o_proj.weight.numel()
    
    print(f"\nAttention Module:")
    print(f"  Q projection: {q_params:,} ({q_params/1e6:.2f}M) - Shape: {list(block.attn.q_proj.weight.shape)}")
    print(f"  K projection: {k_params:,} ({k_params/1e6:.2f}M) - Shape: {list(block.attn.k_proj.weight.shape)}")
    print(f"  V projection: {v_params:,} ({v_params/1e6:.2f}M) - Shape: {list(block.attn.v_proj.weight.shape)}")
    print(f"  O projection: {o_params:,} ({o_params/1e6:.2f}M) - Shape: {list(block.attn.o_proj.weight.shape)}")
    print(f"  Total attention: {attn_params:,} ({attn_params/1e6:.2f}M)")
    
    # FFN (SwiGLU) parameters
    ffn_params = sum(p.numel() for p in block.ffn.parameters())
    gate_params = block.ffn.gate_proj.weight.numel()
    up_params = block.ffn.up_proj.weight.numel()
    down_params = block.ffn.down_proj.weight.numel()
    
    print(f"\nFFN (SwiGLU) Module:")
    print(f"  Gate projection: {gate_params:,} ({gate_params/1e6:.2f}M) - Shape: {list(block.ffn.gate_proj.weight.shape)}")
    print(f"  Up projection: {up_params:,} ({up_params/1e6:.2f}M) - Shape: {list(block.ffn.up_proj.weight.shape)}")
    print(f"  Down projection: {down_params:,} ({down_params/1e6:.2f}M) - Shape: {list(block.ffn.down_proj.weight.shape)}")
    print(f"  Total FFN: {ffn_params:,} ({ffn_params/1e6:.2f}M)")
    
    # LayerNorm parameters
    ln_params = sum(p.numel() for p in block.ln_1.parameters()) + sum(p.numel() for p in block.ln_2.parameters())
    print(f"\nLayerNorm: {ln_params:,} ({ln_params/1e3:.1f}K)")
    
    # Total per block
    block_params = attn_params + ffn_params + ln_params
    print(f"\nTotal per block: {block_params:,} ({block_params/1e6:.2f}M)")
    print(f"Total for {config.n_layer} blocks: {block_params * config.n_layer:,} ({block_params * config.n_layer/1e6:.1f}M)")
    
    # Final layer norm
    final_ln_params = sum(p.numel() for p in model.transformer.ln_f.parameters())
    print(f"\nFinal LayerNorm: {final_ln_params:,} ({final_ln_params/1e3:.1f}K)")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total Model: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Embeddings: {total_embedding_params:,} ({total_embedding_params/1e6:.1f}M) - {total_embedding_params/total_params*100:.1f}%")
    print(f"Transformer Core: {transformer_params:,} ({transformer_params/1e6:.1f}M) - {transformer_params/total_params*100:.1f}%")
    
    print(f"\n{'='*70}")
    print("COMPARISON WITH VOCABULARY SIZES")
    print(f"{'='*70}")
    
    # Calculate what parameters would be with different vocab sizes
    vocab_sizes = [32000, 49152, 65536, 100000, 200000]
    for vocab in vocab_sizes:
        emb_params = vocab * config.n_embd  # Only token embeddings, no position embeddings (RoPE)
        total = transformer_params + emb_params
        print(f"Vocab {vocab:,}: {total/1e6:.1f}M total ({emb_params/1e6:.1f}M embeddings, {emb_params/total*100:.1f}% of model)")
    
    print(f"\n{'='*70}")
    print("GQA SAVINGS")
    print(f"{'='*70}")
    full_attn_params = config.n_embd * config.n_embd * 4  # Q, K, V, O all full size
    gqa_attn_params = (config.n_embd * config.n_embd * 2 +  # Q and O full size
                       config.n_embd * (config.n_kv_heads * config.head_dim) * 2)  # K and V compressed
    savings_per_layer = full_attn_params - gqa_attn_params
    total_savings = savings_per_layer * config.n_layer
    
    print(f"Without GQA (full attention): {full_attn_params/1e6:.2f}M per layer")
    print(f"With GQA (3 KV heads): {gqa_attn_params/1e6:.2f}M per layer")
    print(f"Savings per layer: {savings_per_layer/1e6:.2f}M")
    print(f"Total savings (12 layers): {total_savings/1e6:.1f}M")
    print(f"Model would be {(total_params + total_savings)/1e6:.1f}M without GQA")

if __name__ == "__main__":
    analyze_model_parameters()