"""Profile the MoE model to identify performance bottlenecks."""

import torch
import torch.nn.functional as F
import time
from model_moe import ModelConfig, OptimizedGPT_GLMini_PRMoE
from contextlib import contextmanager

@contextmanager
def timer(name):
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    end = time.time()
    print(f"{name}: {(end - start) * 1000:.2f}ms")

def profile_model():
    # Config
    config = ModelConfig()
    model = OptimizedGPT_GLMini_PRMoE(config).cuda().to(torch.bfloat16)
    
    # Test input
    batch_size = 6
    seq_len = 2048
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    
    # Warmup
    print("Warmup...")
    for _ in range(3):
        with torch.no_grad():
            _, loss = model(x, x)
    
    print("\n=== Profiling Forward Pass ===")
    
    # Profile full forward
    with torch.no_grad():
        with timer("Total Forward"):
            logits, loss = model(x, x)
    
    # Profile individual components
    print("\n=== Component Breakdown ===")
    
    # Test single layer components
    test_input = torch.randn(batch_size, seq_len, config.n_embd).cuda().bfloat16()
    rope_cache = model.rope_cache
    
    # Profile first block (dense)
    if len(model.transformer.h) > 0:
        with torch.no_grad():
            with timer("Dense Block (layer 0)"):
                block_out = model.transformer.h[0](test_input, rope_cache)
    
    # Profile MoE block
    if len(model.transformer.h) > 1:
        with torch.no_grad():
            with timer("MoE Block (layer 1)"):
                block_out = model.transformer.h[1](test_input, rope_cache)
        
        # Profile attention separately
        if hasattr(model.transformer.h[1], 'attn'):
            with torch.no_grad():
                with timer("  - Attention only"):
                    ln_out = model.transformer.h[1].ln_1(test_input)
                    attn_out = model.transformer.h[1].attn(ln_out, rope_cache)
        
        # Profile MoE separately
        if hasattr(model.transformer.h[1], 'ffn'):
            with torch.no_grad():
                with timer("  - MoE FFN only"):
                    ln_out = model.transformer.h[1].ln_2(test_input)
                    moe_out = model.transformer.h[1].ffn(ln_out)
            
            # Profile MoE components
            if hasattr(model.transformer.h[1].ffn, '_route_sigmoid'):
                with torch.no_grad():
                    with timer("    - Router only"):
                        weights, active = model.transformer.h[1].ffn._route_sigmoid(test_input)
            
            if hasattr(model.transformer.h[1].ffn, '_base'):
                with torch.no_grad():
                    with timer("    - Base MLP only"):
                        base_out = model.transformer.h[1].ffn._base(test_input)
    
    # Test masked indexing vs index operations
    print("\n=== Indexing Operations Test ===")
    mask = torch.rand(batch_size, seq_len).cuda() > 0.66  # ~33% selected
    
    with timer("Masked gather x[mask]"):
        selected = test_input[mask]
    
    with timer("Index-based gather"):
        indices = mask.nonzero(as_tuple=True)
        selected2 = test_input[indices]
    
    out_tensor = torch.zeros_like(test_input)
    result = torch.randn_like(selected)
    
    with timer("Masked scatter out[mask] = x"):
        out_tensor[mask] = result
    
    out_tensor2 = torch.zeros_like(test_input)
    with timer("Index-based scatter"):
        out_tensor2[indices] = result
    
    # Memory info
    print(f"\n=== Memory ===")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

if __name__ == "__main__":
    profile_model()