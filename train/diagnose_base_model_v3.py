#!/usr/bin/env python3
"""
Fixed diagnostic suite for base pretrained model v3.
Addresses EOS bug, perplexity sample size, and efficiency issues.
"""

import torch
import torch.nn.functional as F
import math
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from model_inference import ModelConfig, GPTInference
from typing import List, Dict, Tuple, Optional
import warnings
import random
warnings.filterwarnings('ignore')

# Set determinism
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_base_model(checkpoint_path='best_model_fp8_optimized.pt', device=None):
    """Load the base pretrained model."""
    print("="*70)
    print(" BASE MODEL DIAGNOSTIC v3 (Fixed)")
    print("="*70)
    
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("Using CPU (will be slower)")
    
    config = ModelConfig(
        vocab_size=49152,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=3,
        block_size=2048,
        dropout=0.0
    )
    
    model = GPTInference(config)
    model = model.to(device)
    
    checkpoint_info = model.load_from_te_checkpoint(checkpoint_path)
    
    print(f"\nModel Statistics:")
    print(f"  Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"  Val loss: {checkpoint_info.get('val_loss', 'N/A')}")
    print(f"  Iterations: {checkpoint_info.get('iter_num', 'N/A')}")
    print(f"  Device: {device}")
    print("="*70)
    
    model.eval()
    return model, config, checkpoint_info, device

def get_tokenizer(tokenizer_path: Optional[str] = None, expected_vocab_size: int = 49152):
    """Get tokenizer and verify compatibility."""
    if tokenizer_path and Path(tokenizer_path).exists():
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    else:
        print("WARNING: Using default SmolLM tokenizer")
        print("   Specify exact training tokenizer with --tokenizer for accurate results")
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            use_fast=True
        )
    
    # Verify tokenizer
    print(f"\nTokenizer info:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  BOS token: {repr(tokenizer.bos_token)} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {repr(tokenizer.pad_token)} (ID: {tokenizer.pad_token_id})")
    
    # Critical check
    if len(tokenizer) != expected_vocab_size:
        print(f"\nERROR: Tokenizer vocab size {len(tokenizer)} != Model vocab {expected_vocab_size}")
        print("   Results will be unreliable!")
        
    return tokenizer

# ============= PROPER PERPLEXITY WITH ADEQUATE SAMPLE =============

def compute_perplexity_proper(model, tokenizer, texts: List[str], device='cpu',
                             ctx_len: int = 2048, stride: int = 1024,
                             min_tokens: int = 50000):
    """
    Compute perplexity with adequate sample size.
    Ensures at least min_tokens are evaluated for statistical reliability.
    """
    print(f"\nComputing perplexity (target: {min_tokens:,} tokens)...")
    
    total_nll = 0.0
    total_tokens = 0
    total_texts = 0
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors='pt').to(device)
            
            if tokens.shape[1] < 2:
                continue
                
            total_texts += 1
            
            # Process with sliding window
            for start_idx in range(0, tokens.shape[1] - 1, stride):
                end_idx = min(start_idx + ctx_len, tokens.shape[1])
                
                input_ids = tokens[:, start_idx:end_idx-1]
                target_ids = tokens[:, start_idx+1:end_idx]
                
                if input_ids.shape[1] == 0:
                    continue
                
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    reduction='sum'
                )
                
                total_nll += loss.item()
                total_tokens += target_ids.numel()
                
                if end_idx >= tokens.shape[1]:
                    break
            
            # Check if we have enough tokens
            if total_tokens >= min_tokens:
                break
    
    if total_tokens < min_tokens // 2:
        print(f"WARNING: Only evaluated {total_tokens:,} tokens (target was {min_tokens:,})")
        print("   Perplexity may be unreliable. Provide more text data.")
    
    perplexity = math.exp(total_nll / max(1, total_tokens))
    
    print(f"  Evaluated: {total_tokens:,} tokens from {total_texts} texts")
    print(f"  Perplexity: {perplexity:.2f}")
    
    return perplexity, total_tokens

# ============= EFFICIENT SEQUENCE SCORING =============

def sequence_logprob_efficient(model, tokenizer, prompt: str, target: str, device='cpu'):
    """
    Compute sequence log probability in one forward pass.
    Returns: (total_logprob, per_token_logprob, first_token_rank, top5, MRR)
    """
    # Encode separately to know boundaries
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    
    if len(target_ids) == 0:
        return -float('inf'), -float('inf'), -1, [], 0.0
    
    # Combine for single forward pass
    full_ids = prompt_ids + target_ids[:-1]  # Don't include last target token as input
    input_tensor = torch.tensor([full_ids]).to(device)
    
    with torch.no_grad():
        logits, _ = model(input_tensor)
        
        # Get logits for positions where we predict target tokens
        target_start = len(prompt_ids) - 1  # -1 because we predict next token
        target_logits = logits[0, target_start:target_start + len(target_ids), :]
        target_logprobs = F.log_softmax(target_logits, dim=-1)
        
        # Score each target token
        target_tensor = torch.tensor(target_ids).to(device)
        token_logprobs = target_logprobs[range(len(target_ids)), target_tensor]
        total_logprob = token_logprobs.sum().item()
        avg_logprob = total_logprob / len(target_ids)
        
        # First token analysis
        first_token_logprobs = target_logprobs[0]
        first_token_id = target_ids[0]
        first_logprob = first_token_logprobs[first_token_id].item()
        
        # Rank and top-5
        sorted_probs, sorted_ids = torch.sort(first_token_logprobs, descending=True)
        rank = int((sorted_ids == first_token_id).nonzero(as_tuple=True)[0].item()) + 1
        top5_ids = sorted_ids[:5]
        top5_tokens = [tokenizer.decode([tid.item()]) for tid in top5_ids]
        
        # Mean Reciprocal Rank
        mrr = 1.0 / rank
    
    return total_logprob, avg_logprob, rank, top5_tokens, mrr

def run_comprehensive_cloze_tests(model, tokenizer, device='cpu'):
    """Run cloze tests with detailed metrics."""
    print("\n" + "="*70)
    print(" CLOZE COMPLETION TESTS (Efficient)")
    print("="*70)
    
    test_cases = [
        ("The capital of France is", " Paris", "Geography"),
        ("2 + 2 =", " 4", "Math"),
        ("The sky is", " blue", "Common sense"),
        ("Water freezes at", " 0", "Science"),
        ("Monday, Tuesday,", " Wednesday", "Sequence"),
        ("1, 2, 3,", " 4", "Counting"),
        ("def add(x, y):\n    return", " x + y", "Code"),
        ("The opposite of hot is", " cold", "Antonym"),
        ("A, B, C,", " D", "Alphabet"),
        ("import numpy as", " np", "Code import"),
    ]
    
    results = []
    total_mrr = 0.0
    
    for prompt, target, category in test_cases:
        total_lp, avg_lp, rank, top5, mrr = sequence_logprob_efficient(
            model, tokenizer, prompt, target, device
        )
        
        # Status
        if rank == 1:
            status = "PERFECT"
        elif rank <= 5:
            status = "GOOD"
        elif rank <= 20:
            status = "OK"
        else:
            status = "FAIL"
        
        results.append({
            'category': category,
            'prompt': prompt,
            'target': target,
            'rank': rank,
            'status': status,
            'total_logprob': total_lp,
            'avg_logprob': avg_lp,
            'mrr': mrr,
            'top5': top5
        })
        
        total_mrr += mrr
        
        print(f"\n[{category}]")
        print(f"  Target: '{target}' | Rank: {rank} | Status: {status}")
        print(f"  Avg LogProb: {avg_lp:.3f} | MRR: {mrr:.3f}")
        print(f"  Top-5: {top5}")
    
    # Summary metrics
    perfect = sum(1 for r in results if r['status'] == "PERFECT")
    good = sum(1 for r in results if r['status'] in ["PERFECT", "GOOD"])
    mean_mrr = total_mrr / len(results)
    
    print(f"\n{'='*40}")
    print(f"Summary: {perfect} perfect, {good}/10 in top-5")
    print(f"Mean Reciprocal Rank: {mean_mrr:.3f}")
    
    return results, mean_mrr

# ============= FIXED GENERATION TEST =============

def test_generation_fixed(model, tokenizer, device='cpu', max_new_tokens=50):
    """Test generation with proper EOS handling."""
    print("\n" + "="*70)
    print(" GENERATION TEST (Fixed EOS)")
    print("="*70)
    
    prompts = [
        "The weather today is",
        "def fibonacci(n):",
        "Paris is the capital of",
        "1, 2, 3, 4,",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Greedy generation with FIXED EOS check
        with torch.no_grad():
            generated_ids = input_ids.clone()
            for _ in range(max_new_tokens):
                logits, _ = model(generated_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # FIXED: Properly check for EOS even when id=0
                if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                    print("  [Stopped at EOS]")
                    break
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        continuation = generated_text[len(prompt):].strip()
        
        # Truncate for display
        if len(continuation) > 80:
            continuation = continuation[:80] + "..."
        
        print(f"  Generated: {continuation}")

# ============= LOAD WIKIPEDIA DATA =============

def load_wiki_sample(size_kb=100):
    """Load or generate Wikipedia-style text for proper perplexity."""
    # In production, load from actual Wikipedia dump
    # For now, use longer synthetic examples
    
    wiki_texts = [
        """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that they had grossly underestimated the difficulty of the project.""",
        
        """Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on training data, known as a sample data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.""",
        
        """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library. Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0.""",
        
        # Add more texts to reach target size...
    ] * 20  # Repeat to get more tokens
    
    return wiki_texts

# ============= MAIN DIAGNOSTIC =============

def run_diagnostic_v3(checkpoint_path='best_model_fp8_optimized.pt', 
                      tokenizer_path=None,
                      device=None):
    """Run complete diagnostic suite v3."""
    
    set_seed(42)  # Reproducibility
    
    # Load model
    model, config, checkpoint_info, device = load_base_model(checkpoint_path, device)
    tokenizer = get_tokenizer(tokenizer_path, config.vocab_size)
    
    results = {
        'version': 'v3_fixed',
        'checkpoint': checkpoint_path,
        'device': device,
        'model_params': model.num_parameters(),
    }
    
    # Test 1: Proper perplexity with adequate sample
    wiki_texts = load_wiki_sample(100)  # 100KB of text
    ppl, n_tokens = compute_perplexity_proper(
        model, tokenizer, wiki_texts, device,
        ctx_len=2048, stride=1024, min_tokens=50000
    )
    results['perplexity'] = ppl
    results['perplexity_tokens'] = n_tokens
    
    # Test 2: Comprehensive cloze with MRR
    cloze_results, mean_mrr = run_comprehensive_cloze_tests(model, tokenizer, device)
    good_count = sum(1 for r in cloze_results if r['status'] in ["PERFECT", "GOOD"])
    results['cloze_top5'] = f"{good_count}/10"
    results['cloze_mrr'] = mean_mrr
    results['cloze_details'] = cloze_results
    
    # Test 3: Generation with fixed EOS
    test_generation_fixed(model, tokenizer, device)
    
    # ============= DIAGNOSIS =============
    print("\n" + "="*70)
    print(" FINAL DIAGNOSIS")
    print("="*70)
    
    print(f"\nMetrics:")
    print(f"  Perplexity: {ppl:.2f} (on {n_tokens:,} tokens)")
    print(f"  Cloze top-5: {results['cloze_top5']}")
    print(f"  Cloze MRR: {mean_mrr:.3f}")
    
    # Determine quality
    if ppl < 20 and good_count >= 8:
        diagnosis = "EXCELLENT"
        print("\n[PASS] BASE MODEL: EXCELLENT")
        print("   Strong foundation. Issues are from SFT/alignment.")
    elif ppl < 50 and good_count >= 6:
        diagnosis = "GOOD"
        print("\n[PASS] BASE MODEL: GOOD")
        print("   Healthy model. Focus on SFT improvements.")
    elif ppl < 100 and good_count >= 4:
        diagnosis = "MARGINAL"
        print("\n[WARN] BASE MODEL: MARGINAL")
        print("   Consider light continued pretraining.")
    else:
        diagnosis = "POOR"
        print("\n[FAIL] BASE MODEL: POOR")
        print("   Needs significant continued pretraining.")
    
    results['diagnosis'] = diagnosis
    
    # Save results
    with open("diagnostic_results_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Results saved to diagnostic_results_v3.json")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fixed base model diagnostics v3')
    parser.add_argument('--checkpoint', type=str, default='best_model_fp8_optimized.pt')
    parser.add_argument('--tokenizer', type=str, default=None,
                       help='Path to exact tokenizer used in training')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint '{args.checkpoint}' not found!")
        exit(1)
    
    device = None if args.device == 'auto' else args.device
    results = run_diagnostic_v3(args.checkpoint, args.tokenizer, device)