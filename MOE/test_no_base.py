"""Test performance without base MLP in PR-MoE."""

import torch
import time
from model_moe import ModelConfig, OptimizedGPT_GLMini_PRMoE

# Monkey-patch to disable base MLP
def forward_no_base(self, x):
    """Forward without base MLP computation."""
    B, T, C = x.shape
    # base_out = self._base(x)  # SKIP THIS!
    base_out = 0  # No base computation
    
    if self.cfg.router_type == "sigmoid":
        weights, active = self._route_sigmoid(x)
    else:
        gate = torch.nn.functional.softmax(self.router(x), dim=-1)
        k = 1
        topv, topi = torch.topk(gate, k=k, dim=-1)
        weights = torch.zeros_like(gate).scatter_(-1, topi, topv)
        active = (weights > 0).to(weights.dtype)
    
    # Expert computation (same as before)
    x_flat = x.view(B * T, C)
    expert_out = torch.zeros(B * T, C, dtype=x.dtype, device=x.device)
    active_flat = active.view(B * T, -1)
    weights_flat = weights.view(B * T, -1)
    
    for e, expert in enumerate(self.experts):
        expert_indices = (active_flat[:, e] > 0).nonzero(as_tuple=True)[0]
        if expert_indices.numel() == 0:
            continue
        
        expert_tokens = x_flat.index_select(0, expert_indices)
        expert_weights = weights_flat[expert_indices, e].unsqueeze(-1)
        
        # Process expert
        gate_out = expert["gate"](expert_tokens)
        up_out = expert["up"](expert_tokens)
        h = torch.nn.functional.silu(gate_out) * up_out
        h = expert["down"](h)
        
        expert_out.index_add_(0, expert_indices, expert_weights * h)
    
    expert_out = expert_out.view(B, T, C)
    return x + base_out + expert_out  # base_out is 0

def test_no_base():
    config = ModelConfig()
    config.use_mtp = False  # Disable MTP for cleaner test
    model = OptimizedGPT_GLMini_PRMoE(config).cuda().bfloat16()
    
    # Patch the forward method
    from model_moe import PyramidResidualMoE
    original_forward = PyramidResidualMoE.forward
    PyramidResidualMoE.forward = forward_no_base
    
    batch_size = 8
    seq_len = 2048
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    
    print("Testing WITH base MLP...")
    PyramidResidualMoE.forward = original_forward
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _, _ = model(x, x)
    torch.cuda.synchronize()
    time_with_base = (time.time() - start) / 10 * 1000
    
    print("Testing WITHOUT base MLP...")
    PyramidResidualMoE.forward = forward_no_base
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _, _ = model(x, x)
    torch.cuda.synchronize()
    time_without_base = (time.time() - start) / 10 * 1000
    
    print(f"\n=== Results ===")
    print(f"WITH base MLP: {time_with_base:.1f}ms")
    print(f"WITHOUT base MLP: {time_without_base:.1f}ms")
    print(f"Speedup: {time_with_base/time_without_base:.2f}x")
    print(f"\nExpected tokens/sec improvement: {time_with_base/time_without_base:.2f}x")

if __name__ == "__main__":
    test_no_base()