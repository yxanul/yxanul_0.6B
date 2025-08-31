# fp8_linear_smoke.py
import os
os.environ.setdefault("NVTE_GEMM_WORKSPACE_SIZE", str(128*1024*1024))
os.environ.setdefault("NVTE_DEBUG","1")
os.environ.setdefault("NVTE_DEBUG_LEVEL","1")

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
try:
    from transformer_engine.common.recipe import MXFP8BlockScaling
    recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)  # Blackwell path
except Exception:
    recipe = DelayedScaling(fp8_format=Format.HYBRID)   # fallback

torch.cuda.set_device(0)
torch.set_float32_matmul_precision('high')

B,T,C = 16, 2048, 768      # M = B*T = 32768 (well-aligned)
HID   = 2048               # multiple-of-64
x = torch.randn(B,T,C, device='cuda', dtype=torch.bfloat16, requires_grad=True)

l1 = te.Linear(C, HID, bias=False, params_dtype=torch.bfloat16).cuda()
l2 = te.Linear(HID, C,  bias=False, params_dtype=torch.bfloat16).cuda()

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = l2(torch.nn.functional.silu(l1(x)))
loss = (y.float()**2).mean()
loss.backward()
print("✓ FP8 smoke test passed:", float(loss))
