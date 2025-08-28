#!/usr/bin/env python3
"""
Analyze FP8 usage in the model and identify bottlenecks.
"""

import torch
import torch.nn as nn
from model_te import TEModelConfig, TETransformerGPT

def analyze_model_layers(model):
    """Analyze which layers use FP8 vs BF16."""
    
    total_params = 0
    fp8_params = 0
    bf16_params = 0
    fp32_params = 0
    
    layer_stats = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            param_count = module.weight.numel()
            total_params += param_count
            
            # Check layer type
            layer_type = type(module).__name__
            module_path = type(module).__module__
            
            # Determine if FP8 capable
            is_fp8 = 'transformer_engine' in module_path
            
            if is_fp8:
                fp8_params += param_count
                precision = "FP8-capable"
            elif isinstance(module, nn.Embedding):
                bf16_params += param_count
                precision = "BF16 (Embedding)"
            elif isinstance(module, nn.Linear):
                bf16_params += param_count  
                precision = "BF16 (Linear)"
            else:
                bf16_params += param_count
                precision = "BF16"
            
            layer_stats.append({
                'name': name,
                'type': layer_type,
                'params': param_count,
                'params_m': param_count / 1e6,
                'precision': precision,
                'percentage': (param_count / 113e6) * 100
            })
    
    # Print analysis
    print("=" * 70)
    print("FP8 UTILIZATION ANALYSIS")
    print("=" * 70)
    print(f"\nTotal parameters: {total_params/1e6:.1f}M")
    print(f"FP8-capable params: {fp8_params/1e6:.1f}M ({fp8_params/total_params*100:.1f}%)")
    print(f"BF16-only params: {bf16_params/1e6:.1f}M ({bf16_params/total_params*100:.1f}%)")
    
    print("\n" + "-" * 70)
    print("LAYER BREAKDOWN")
    print("-" * 70)
    print(f"{'Layer':<40} {'Type':<15} {'Params':<10} {'Precision':<15}")
    print("-" * 70)
    
    for stat in sorted(layer_stats, key=lambda x: -x['params']):
        if stat['params'] > 10000:  # Only show significant layers
            print(f"{stat['name'][:39]:<40} {stat['type']:<15} {stat['params_m']:>6.2f}M {stat['precision']:<15}")
    
    print("\n" + "-" * 70)
    print("BOTTLENECK ANALYSIS")
    print("-" * 70)
    
    # Identify bottlenecks
    embedding_params = sum(s['params'] for s in layer_stats if 'Embedding' in s['type'])
    lm_head_params = sum(s['params'] for s in layer_stats if 'lm_head' in s['name'])
    
    print(f"\n1. Embeddings + LM Head: {(embedding_params + lm_head_params)/1e6:.1f}M params ({(embedding_params + lm_head_params)/total_params*100:.1f}%)")
    print(f"   - These are in BF16 (not FP8)")
    print(f"   - Weight tying reduces impact but still computed in BF16")
    
    # Check for non-TE linear layers
    non_te_linear = [s for s in layer_stats if s['type'] == 'Linear' and 'BF16' in s['precision']]
    if non_te_linear:
        non_te_params = sum(s['params'] for s in non_te_linear)
        print(f"\n2. Non-TE Linear layers: {non_te_params/1e6:.1f}M params ({non_te_params/total_params*100:.1f}%)")
        print(f"   - LM head uses nn.Linear for weight tying")
    
    # Memory bandwidth analysis
    print("\n3. Memory Bandwidth Bottlenecks:")
    print(f"   - Batch size: 64 (large memory transfers)")
    print(f"   - Sequence length: 128 tokens")
    print(f"   - Gradient accumulation: 16 steps")
    print(f"   - Effective batch: 1024 sequences per optimizer step")
    
    # FP8 overhead analysis  
    print("\n4. FP8 Overhead:")
    print(f"   - Format: HYBRID (E4M3 forward, E5M2 backward)")
    print(f"   - Calibration steps: 10")
    print(f"   - Amax history: 16 steps")
    print(f"   - Tensor alignment padding overhead")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUGGESTIONS")
    print("=" * 70)
    
    print("\n1. Use E4M3 format instead of HYBRID:")
    print("   - HYBRID uses E5M2 for gradients (less speedup)")
    print("   - Pure E4M3 gives more aggressive quantization")
    
    print("\n2. Reduce batch size to decrease memory bandwidth:")
    print("   - Try batch_size=32 or 16")
    print("   - Memory bandwidth may be the bottleneck, not compute")
    
    print("\n3. Convert ALL Linear layers to te.Linear:")
    print("   - LM head currently uses nn.Linear")
    print("   - Consider not tying weights to enable FP8 for output")
    
    print("\n4. Profile with Nsight to identify actual bottlenecks:")
    print("   - Use: nsys profile -o profile python train_te_fair.py")
    print("   - Check if kernels are actually using Tensor Cores")
    
    print("\n5. Check if FP8 is actually engaged:")
    print("   - Set NVTE_DEBUG=1 environment variable")
    print("   - Look for FP8 kernel launches in output")

if __name__ == "__main__":
    # Create model
    config = TEModelConfig(
        vocab_size=50256,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=12,
        block_size=128,
        dropout=0.0,
        bias=False
    )
    config.ffn_hidden_size = 2048
    
    model = TETransformerGPT(config)
    
    # Analyze
    analyze_model_layers(model)
    
    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 70)
    print("\n1. Test with pure E4M3 format:")
    print("   python train_te_fair.py --fp8_format E4M3")
    print("\n2. Profile with environment variable:")
    print("   NVTE_DEBUG=1 python train_te_fair.py --max_iters 10")
    print("\n3. Try smaller batch size:")
    print("   Modify batch_size in train_te_fair.py from 64 to 16")