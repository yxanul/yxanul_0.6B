#!/usr/bin/env python3
"""
Test script for the SFT (instruction-tuned) model.
This loads the fine-tuned model and tests instruction-following capabilities.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer
from model_inference import ModelConfig, GPTInference
import time

def load_sft_model(checkpoint_path='checkpoints_sft/best_sft_model.pt'):
    """Load the SFT trained model."""
    print("="*60)
    print("Loading SFT Model (Instruction-Tuned)")
    print("="*60)
    
    # Create config
    config = ModelConfig(
        vocab_size=49152,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_kv_heads=3,
        block_size=2048,
        dropout=0.0  # No dropout for inference
    )
    
    # Create model
    model = GPTInference(config)
    
    # Load checkpoint
    checkpoint_info = model.load_from_te_checkpoint(checkpoint_path)
    
    # Model info
    print(f"\nModel Statistics:")
    print(f"  Parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"  Architecture: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
    print(f"  Training: SFT on instruction data")
    print(f"  Format: User/Assistant conversation")
    print("="*60)
    
    model.eval()
    return model, config

def get_tokenizer():
    """Get the SmolLM tokenizer used for training."""
    print("Loading SmolLM2-135M tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        use_fast=True
    )
    print(f"  Vocab size: {len(tokenizer)}")
    return tokenizer

@torch.no_grad()
def generate_response(model, tokenizer, user_input, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.85, repetition_penalty=1.1):
    """Generate response in instruction format."""
    # Format as instruction
    prompt = f"User: {user_input}\nAssistant:"
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
    
    print(f"\n👤 User: {user_input}")
    print("🤖 Assistant: ", end="", flush=True)
    
    start_time = time.time()
    
    # Generate tokens
    generated_ids = input_ids
    response_start = len(generated_ids[0])
    
    for i in range(max_new_tokens):
        # Get logits
        with torch.no_grad():
            logits, _ = model(generated_ids)
        
        # Get last token logits
        logits = logits[:, -1, :] / temperature
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()[response_start:]):  # Only penalize response tokens
                logits[0, token_id] /= repetition_penalty
        
        # Apply top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Decode token to check for stop
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
        
        # Stop if we see "User:" indicating next turn
        if "User:" in tokenizer.decode(generated_ids[0][response_start:], skip_special_tokens=False) + token_text:
            break
        
        # Append
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # Print progress dots sparingly
        if i % 10 == 0:
            print(".", end="", flush=True)
    
    print(" Done!")
    
    # Decode full generation
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Extract just the assistant response
    if "Assistant:" in full_text:
        response = full_text.split("Assistant:")[-1].strip()
        # Remove any trailing "User:" if present
        if "User:" in response:
            response = response.split("User:")[0].strip()
    else:
        response = full_text[len(f"User: {user_input}"):].strip()
    
    elapsed = time.time() - start_time
    tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
    print(f"⏱️  Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.1f} tok/s)")
    
    return response

def run_instruction_tests(model, tokenizer):
    """Run comprehensive instruction-following tests."""
    print("\n" + "="*60)
    print("INSTRUCTION-FOLLOWING TEST SUITE")
    print("Testing SFT Model Quality")
    print("="*60)
    
    test_cases = [
        # Basic Q&A
        {
            "input": "What is the capital of France?",
            "category": "Factual Q&A",
            "temperature": 0.5
        },
        
        # Explanation
        {
            "input": "Explain photosynthesis in simple terms.",
            "category": "Explanation",
            "temperature": 0.7
        },
        
        # Creative writing
        {
            "input": "Write a haiku about artificial intelligence.",
            "category": "Creative",
            "temperature": 0.8
        },
        
        # Math/Reasoning
        {
            "input": "If I have 3 apples and buy 5 more, then give 2 to a friend, how many do I have?",
            "category": "Math Reasoning",
            "temperature": 0.5
        },
        
        # Code generation
        {
            "input": "Write a Python function to reverse a string.",
            "category": "Code Generation",
            "temperature": 0.5
        },
        
        # Summarization
        {
            "input": "Summarize the following in one sentence: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            "category": "Summarization",
            "temperature": 0.6
        },
        
        # List generation
        {
            "input": "List 3 benefits of regular exercise.",
            "category": "List Generation",
            "temperature": 0.7
        },
        
        # Translation/Transformation
        {
            "input": "Convert this to a more formal tone: 'Hey, can you help me out with this thing?'",
            "category": "Style Transfer",
            "temperature": 0.6
        },
        
        # Instruction following
        {
            "input": "Write the word 'hello' in exactly 5 different ways.",
            "category": "Instruction Following",
            "temperature": 0.7
        },
        
        # Open-ended
        {
            "input": "What are your thoughts on the future of technology?",
            "category": "Open Discussion",
            "temperature": 0.8
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] Category: {test['category']}")
        print("-"*60)
        
        response = generate_response(
            model, tokenizer,
            test["input"],
            max_new_tokens=150,
            temperature=test.get("temperature", 0.7),
            top_k=40,
            top_p=0.85,
            repetition_penalty=1.1
        )
        
        print(f"\n💬 Response:\n{response}")
        print("-"*60)
        
        # Brief pause between tests
        time.sleep(0.5)
    
    print("\n" + "="*60)
    print("INSTRUCTION TEST SUITE COMPLETE!")
    print("Quality Checklist:")
    print("  ✅ Response Relevance: Does it answer the question?")
    print("  ✅ Instruction Following: Does it follow specific instructions?")
    print("  ✅ Coherence: Are responses well-structured?")
    print("  ✅ Knowledge: Are facts reasonable?")
    print("  ✅ Format: Does it maintain User/Assistant format?")
    print("="*60)

def interactive_chat(model, tokenizer):
    """Interactive chat mode."""
    print("\n" + "="*60)
    print("INTERACTIVE CHAT MODE")
    print("Type 'quit' to exit, 'help' for options")
    print("="*60)
    
    settings = {
        'temperature': 0.7,
        'top_k': 40,
        'top_p': 0.85,
        'repetition_penalty': 1.1,
        'max_tokens': 150
    }
    
    print("\n🎯 This is an instruction-tuned model.")
    print("   Just type your questions or requests naturally!")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'help':
            print("\nCommands:")
            print("  quit - Exit")
            print("  help - Show this help")
            print("  temp=X - Set temperature (0.1-2.0)")
            print("  max=X - Set max tokens (10-500)")
            print("\nJust type naturally to chat with the model!")
            continue
        elif user_input.startswith('temp='):
            try:
                settings['temperature'] = float(user_input.split('=')[1])
                print(f"✅ Temperature set to {settings['temperature']}")
            except:
                print("❌ Invalid temperature")
            continue
        elif user_input.startswith('max='):
            try:
                settings['max_tokens'] = int(user_input.split('=')[1])
                print(f"✅ Max tokens set to {settings['max_tokens']}")
            except:
                print("❌ Invalid max tokens")
            continue
        
        if user_input:
            response = generate_response(
                model, tokenizer, user_input,
                max_new_tokens=settings['max_tokens'],
                temperature=settings['temperature'],
                top_k=settings['top_k'],
                top_p=settings['top_p'],
                repetition_penalty=settings['repetition_penalty']
            )
            print(f"\n🤖 Assistant:\n{response}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test the SFT instruction-tuned model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_sft/best_sft_model.pt',
                       help='Path to SFT model checkpoint')
    parser.add_argument('--mode', type=str, choices=['test', 'chat', 'single'],
                       default='test', help='Test mode')
    parser.add_argument('--input', type=str, help='Single instruction to test')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=150)
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint '{args.checkpoint}' not found!")
        print(f"   Looking for: {args.checkpoint}")
        print(f"   Make sure SFT training has completed and saved the model.")
        return
    
    # Load model and tokenizer
    model, config = load_sft_model(args.checkpoint)
    tokenizer = get_tokenizer()
    
    print("\n🚀 SFT Model Ready!")
    print(f"   Type: Instruction-tuned")
    print(f"   Base: Elite 2.218 val loss model")
    print(f"   Training: 200k conversations")
    
    # Run appropriate mode
    if args.mode == 'test':
        run_instruction_tests(model, tokenizer)
    elif args.mode == 'chat':
        interactive_chat(model, tokenizer)
    elif args.mode == 'single':
        if not args.input:
            print("❌ Please provide an instruction with --input")
            return
        response = generate_response(
            model, tokenizer, args.input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"\n💬 Response:\n{response}")

if __name__ == "__main__":
    main()