#!/usr/bin/env python3
"""
Compare GPT-2 tokenizer vs SuperBPE tokenizer.
Measures token count, compression ratio, and bytes per token.
"""

import tiktoken
from transformers import AutoTokenizer

def analyze_tokenization(text, tokenizer_name, tokens):
    """Analyze tokenization metrics."""
    # Calculate metrics
    num_tokens = len(tokens)
    num_bytes = len(text.encode('utf-8'))
    num_chars = len(text)
    
    bytes_per_token = num_bytes / num_tokens if num_tokens > 0 else 0
    chars_per_token = num_chars / num_tokens if num_tokens > 0 else 0
    
    return {
        'tokenizer': tokenizer_name,
        'num_tokens': num_tokens,
        'num_bytes': num_bytes,
        'num_chars': num_chars,
        'bytes_per_token': bytes_per_token,
        'chars_per_token': chars_per_token,
        'compression_ratio': num_bytes / (num_tokens * 2) if num_tokens > 0 else 0  # Assuming 2 bytes per token ID
    }

def main():
    # Test text from TinyStories
    test_text = """One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt.

Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt."

Together, they shared the needle and sewed the button on Lily's shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together."""

    print("="*70)
    print("TOKENIZER COMPARISON: GPT-2 vs SuperBPE")
    print("="*70)
    
    print(f"\nTest text length: {len(test_text)} characters")
    print(f"Test text bytes: {len(test_text.encode('utf-8'))} bytes")
    print("-"*70)
    
    # 1. GPT-2 Tokenizer (tiktoken)
    print("\n1. GPT-2 Tokenizer (tiktoken)")
    print("-"*40)
    try:
        gpt2_enc = tiktoken.get_encoding("gpt2")
        gpt2_tokens = gpt2_enc.encode(test_text)
        gpt2_stats = analyze_tokenization(test_text, "GPT-2", gpt2_tokens)
        
        print(f"  Tokens: {gpt2_stats['num_tokens']}")
        print(f"  Bytes per token: {gpt2_stats['bytes_per_token']:.2f}")
        print(f"  Chars per token: {gpt2_stats['chars_per_token']:.2f}")
        
        # Show first 20 tokens
        print(f"\n  First 20 tokens: {gpt2_tokens[:20]}")
        decoded_sample = [gpt2_enc.decode([t]) for t in gpt2_tokens[:20]]
        print(f"  Decoded: {decoded_sample}")
        
    except Exception as e:
        print(f"  Error: {e}")
        gpt2_stats = None
    
    # 2. OLMo2 SuperBPE-180k Tokenizer (the correct one)
    print("\n2. OLMo2 SuperBPE-180k Tokenizer")
    print("-"*40)
    try:
        # Load the EXACT tokenizer specified with use_fast=True
        olmo_tokenizer = AutoTokenizer.from_pretrained(
            "UW/OLMo2-8B-SuperBPE-180k",
            use_fast=True
        )
        olmo_tokens = olmo_tokenizer.encode(test_text, add_special_tokens=False)
        olmo_stats = analyze_tokenization(test_text, "SuperBPE-180k", olmo_tokens)
        
        print(f"  Tokens: {olmo_stats['num_tokens']}")
        print(f"  Bytes per token: {olmo_stats['bytes_per_token']:.2f}")
        print(f"  Chars per token: {olmo_stats['chars_per_token']:.2f}")
        print(f"  Vocabulary size: {len(olmo_tokenizer)}")
        
        # Show first 20 tokens
        print(f"\n  First 20 tokens: {olmo_tokens[:20]}")
        decoded_sample = [olmo_tokenizer.decode([t]) for t in olmo_tokens[:20]]
        print(f"  Decoded: {decoded_sample}")
        
    except Exception as e:
        print(f"  Error loading UW/OLMo2-8B-SuperBPE-180k: {e}")
        print(f"  Trying fallback...")
        try:
            # Try alternative SuperBPE tokenizer
            olmo_tokenizer = AutoTokenizer.from_pretrained(
                "UW/OLMo2-8B-SuperBPE-t80k",  # Try the t80k variant
                use_fast=True
            )
            olmo_tokens = olmo_tokenizer.encode(test_text, add_special_tokens=False)
            olmo_stats = analyze_tokenization(test_text, "SuperBPE-t80k", olmo_tokens)
            
            print(f"  Using SuperBPE-t80k variant")
            print(f"  Tokens: {olmo_stats['num_tokens']}")
            print(f"  Bytes per token: {olmo_stats['bytes_per_token']:.2f}")
            print(f"  Chars per token: {olmo_stats['chars_per_token']:.2f}")
            print(f"  Vocabulary size: {len(olmo_tokenizer)}")
            
            # Show first 20 tokens
            print(f"\n  First 20 tokens: {olmo_tokens[:20]}")
            decoded_sample = [olmo_tokenizer.decode([t]) for t in olmo_tokens[:20]]
            print(f"  Decoded: {decoded_sample}")
        except Exception as e2:
            print(f"  Error: {e2}")
            olmo_stats = None
    
    # 3. Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    if gpt2_stats and olmo_stats:
        print(f"\n{'Metric':<25} {'GPT-2':<15} {'SuperBPE/Alt':<15} {'Difference':<15}")
        print("-"*70)
        
        # Token count
        token_diff = olmo_stats['num_tokens'] - gpt2_stats['num_tokens']
        token_pct = (token_diff / gpt2_stats['num_tokens']) * 100
        print(f"{'Token count':<25} {gpt2_stats['num_tokens']:<15} {olmo_stats['num_tokens']:<15} "
              f"{token_diff:+d} ({token_pct:+.1f}%)")
        
        # Bytes per token
        bpt_diff = olmo_stats['bytes_per_token'] - gpt2_stats['bytes_per_token']
        bpt_pct = (bpt_diff / gpt2_stats['bytes_per_token']) * 100
        print(f"{'Bytes per token':<25} {gpt2_stats['bytes_per_token']:<15.2f} "
              f"{olmo_stats['bytes_per_token']:<15.2f} {bpt_diff:+.2f} ({bpt_pct:+.1f}%)")
        
        # Chars per token
        cpt_diff = olmo_stats['chars_per_token'] - gpt2_stats['chars_per_token']
        cpt_pct = (cpt_diff / gpt2_stats['chars_per_token']) * 100
        print(f"{'Chars per token':<25} {gpt2_stats['chars_per_token']:<15.2f} "
              f"{olmo_stats['chars_per_token']:<15.2f} {cpt_diff:+.2f} ({cpt_pct:+.1f}%)")
        
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        if token_pct < 0:
            print(f"\n[GOOD] SuperBPE/Alternative uses {abs(token_pct):.1f}% FEWER tokens!")
            print(f"  This means {abs(token_pct):.1f}% faster processing")
            print(f"  And {abs(token_pct):.1f}% longer effective context")
        else:
            print(f"\n[INFO] SuperBPE/Alternative uses {token_pct:.1f}% MORE tokens")
            print(f"  GPT-2 tokenizer is more efficient for this text")
        
        print(f"\nEfficiency metrics:")
        print(f"  GPT-2: {gpt2_stats['chars_per_token']:.2f} chars/token")
        print(f"  SuperBPE/Alt: {olmo_stats['chars_per_token']:.2f} chars/token")
        
        if olmo_stats['chars_per_token'] > gpt2_stats['chars_per_token']:
            improvement = ((olmo_stats['chars_per_token'] / gpt2_stats['chars_per_token']) - 1) * 100
            print(f"  -> SuperBPE/Alt is {improvement:.1f}% more efficient per token")
        
    else:
        print("Could not compare - one or both tokenizers failed to load")
    
    # Test specific phrases
    print("\n" + "="*70)
    print("PHRASE-LEVEL COMPARISON")
    print("="*70)
    
    test_phrases = [
        "One day, a little girl named Lily",
        "She knew it was difficult",
        "Can you share it with me",
        "They both felt happy",
    ]
    
    if gpt2_stats:
        print("\nPhrase tokenization comparison:")
        print("-"*70)
        for phrase in test_phrases:
            gpt2_phrase_tokens = gpt2_enc.encode(phrase)
            print(f"\nPhrase: '{phrase}'")
            print(f"  GPT-2: {len(gpt2_phrase_tokens)} tokens")
            
            if olmo_stats:
                try:
                    olmo_phrase_tokens = olmo_tokenizer.encode(phrase, add_special_tokens=False)
                    print(f"  SuperBPE/Alt: {len(olmo_phrase_tokens)} tokens")
                    diff = len(olmo_phrase_tokens) - len(gpt2_phrase_tokens)
                    print(f"  Difference: {diff:+d} tokens")
                except:
                    pass

if __name__ == "__main__":
    print("Testing tokenizer efficiency...\n")
    main()
    print("\nNote: SuperBPE tokenizers are optimized for specific domains.")
    print("The t80k variant should show 20-30% token reduction on typical English text.")