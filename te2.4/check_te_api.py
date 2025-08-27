#!/usr/bin/env python3
"""Check TransformerEngine API for fp8_model_init"""

import sys

try:
    import transformer_engine as te
    import transformer_engine.pytorch as te_pytorch
    print(f"TransformerEngine version: {te.__version__}")
    
    # Check if fp8_model_init exists and what it is
    if hasattr(te_pytorch, 'fp8_model_init'):
        import inspect
        func = te_pytorch.fp8_model_init
        print(f"\nfp8_model_init found!")
        print(f"Type: {type(func)}")
        
        # Get signature
        sig = inspect.signature(func)
        print(f"Signature: {sig}")
        
        # Check if it's a context manager
        if hasattr(func, '__enter__') and hasattr(func, '__exit__'):
            print("It's a context manager")
        else:
            print("It's a function")
            
        # Get docstring
        if func.__doc__:
            print(f"\nDocstring:\n{func.__doc__[:500]}...")
    else:
        print("fp8_model_init not found in te_pytorch")
        
    # List available functions
    print("\n\nAvailable FP8-related functions:")
    for name in dir(te_pytorch):
        if 'fp8' in name.lower():
            print(f"  - {name}")
            
except ImportError as e:
    print(f"TransformerEngine not available: {e}")
    sys.exit(1)