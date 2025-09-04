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
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: Running on CPU - will be slow!")
    
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
    
    # Move to same device as model
    if next(model.parameters()).is_cuda:
        input_ids = input_ids.cuda()
    
    print(f"\nUser: {user_input}")
    print("Assistant: ", end="", flush=True)
    
    start_time = time.time()
    
    # Generate tokens
    generated_ids = input_ids
    response_start = len(generated_ids[0])
    
    for i in range(max_new_tokens):
        # Get logits
        with torch.no_grad():
            logits, _ = model(generated_ids)
        
        # Get last token logits
        if temperature > 0:
            logits = logits[:, -1, :] / temperature
        else:
            # Greedy decoding (temperature=0)
            logits = logits[:, -1, :]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated_ids[0].tolist()[response_start:]):  # Only penalize response tokens
                logits[0, token_id] /= repetition_penalty
        
        # Apply top-k filtering (skip if top_k=0 for greedy)
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
        
        # Sample or greedy decode
        if temperature == 0 or top_k == 0:
            # Greedy decoding: take argmax
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
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
        
        # Check for EOS (fixed for ID=0)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
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
    print(f"[TIME] Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.1f} tok/s)")
    
    return response

def run_instruction_tests(model, tokenizer):
    """Run comprehensive instruction-following tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION SUITE")
    print("Testing with Low Temperature for Deterministic Output")
    print("="*60)
    
    test_cases = [
        # === KNOWLEDGE & TRIVIA ===
        {
            "input": "What is the capital of France?",
            "category": "Geography",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        {
            "input": "Who wrote Romeo and Juliet?",
            "category": "Literature",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        {
            "input": "What year did World War II end?",
            "category": "History",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        {
            "input": "What is the chemical symbol for gold?",
            "category": "Science",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        {
            "input": "How many planets are in our solar system?",
            "category": "Astronomy",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        
        # === MATH PROBLEMS (Varying Difficulty) ===
        {
            "input": "What is 7 + 8?",
            "category": "Math: Basic Addition",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        {
            "input": "Calculate 15 * 4",
            "category": "Math: Multiplication",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        {
            "input": "If I have 3 apples and buy 5 more, then give 2 to a friend, how many do I have?",
            "category": "Math: Word Problem Easy",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 32,
            "repetition_penalty": 1.0
        },
        {
            "input": "A train travels 60 miles in 1.5 hours. What is its average speed?",
            "category": "Math: Word Problem Medium",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 32,
            "repetition_penalty": 1.0
        },
        {
            "input": "If x + 3 = 10, what is x?",
            "category": "Math: Simple Algebra",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        
        # === INSTRUCTION FOLLOWING ===
        {
            "input": "Write exactly 3 words.",
            "category": "Instruction: Count Constraint",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 8,
            "repetition_penalty": 1.0
        },
        {
            "input": "Write the word 'hello' in ALL CAPS.",
            "category": "Instruction: Format",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 8,
            "repetition_penalty": 1.0
        },
        {
            "input": "List the numbers from 1 to 5.",
            "category": "Instruction: Sequence",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        {
            "input": "Answer with only 'yes' or 'no': Is the sky blue?",
            "category": "Instruction: Binary Choice",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 4,
            "repetition_penalty": 1.0
        },
        {
            "input": "Complete this sentence: The cat sat on the ___",
            "category": "Instruction: Fill Blank",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 8,
            "repetition_penalty": 1.0
        },
        
        # === REASONING & COMPREHENSION ===
        {
            "input": "Which is bigger: an elephant or a mouse?",
            "category": "Reasoning: Comparison",
            "temperature": 0.1
        },
        {
            "input": "If all birds can fly, and a penguin is a bird, can a penguin fly?",
            "category": "Reasoning: Logic",
            "temperature": 0.1
        },
        {
            "input": "It's raining. Should I take an umbrella? Answer yes or no and explain why.",
            "category": "Reasoning: Common Sense",
            "temperature": 0.1
        },
        {
            "input": "What comes next in this pattern: 2, 4, 6, 8, ?",
            "category": "Reasoning: Pattern",
            "temperature": 0.1
        },
        {
            "input": "If today is Monday, what day will it be in 3 days?",
            "category": "Reasoning: Temporal",
            "temperature": 0.1
        },
        
        # === CODE & TECHNICAL ===
        {
            "input": "Write a Python print statement that outputs 'Hello World'",
            "category": "Code: Basic",
            "temperature": 0.1
        },
        {
            "input": "What is a variable in programming?",
            "category": "Code: Concept",
            "temperature": 0.1
        },
        {
            "input": "Write a Python function to add two numbers.",
            "category": "Code: Function",
            "temperature": 0.1
        },
        
        # === LANGUAGE TASKS ===
        {
            "input": "What is the opposite of 'hot'?",
            "category": "Language: Antonym",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 8,
            "repetition_penalty": 1.0
        },
        {
            "input": "What is the plural of 'mouse'?",
            "category": "Language: Grammar",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 8,
            "repetition_penalty": 1.0
        },
        {
            "input": "Translate 'hello' to Spanish.",
            "category": "Language: Translation",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 8,
            "repetition_penalty": 1.0
        },
        {
            "input": "Fix this sentence: 'Me go store yesterday'",
            "category": "Language: Correction",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "repetition_penalty": 1.0
        },
        
        # === EXPLANATION TASKS ===
        {
            "input": "Explain gravity in one sentence.",
            "category": "Explanation: Science",
            "temperature": 0.1
        },
        {
            "input": "What is photosynthesis?",
            "category": "Explanation: Biology",
            "temperature": 0.1
        },
        {
            "input": "Why is the sky blue?",
            "category": "Explanation: Physics",
            "temperature": 0.1
        },
        
        # === CREATIVE (Higher temp for variety) ===
        {
            "input": "Write a haiku about artificial intelligence.",
            "category": "Creative: Poetry",
            "temperature": 0.5
        },
        {
            "input": "Generate a creative name for a robot.",
            "category": "Creative: Naming",
            "temperature": 0.5
        },
        
        # === CLASSIFICATION ===
        {
            "input": "Is 'happy' a noun or an adjective?",
            "category": "Classification: Grammar",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 8,
            "repetition_penalty": 1.0
        },
        {
            "input": "Classify this sentence as positive or negative: 'I love this movie!'",
            "category": "Classification: Sentiment",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 8,
            "repetition_penalty": 1.0
        },
        {
            "input": "Is a tomato a fruit or a vegetable?",
            "category": "Classification: Knowledge",
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_tokens": 8,
            "repetition_penalty": 1.0
        }
    ]
    
    # Track results for summary
    results_by_category = {}
    all_results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] Category: {test['category']}")
        print("-"*60)
        
        response = generate_response(
            model, tokenizer,
            test["input"],
            max_new_tokens=test.get("max_tokens", 150),
            temperature=test.get("temperature", 0.7),
            top_k=test.get("top_k", 40),
            top_p=test.get("top_p", 0.85),
            repetition_penalty=test.get("repetition_penalty", 1.1)
        )
        
        print(f"\n[RESPONSE]\n{response}")
        print("-"*60)
        
        # Store result for analysis
        result = {
            "category": test["category"],
            "input": test["input"],
            "response": response,
            "temperature": test.get("temperature", 0.7)
        }
        all_results.append(result)
        
        # Group by category prefix
        category_prefix = test["category"].split(":")[0]
        if category_prefix not in results_by_category:
            results_by_category[category_prefix] = []
        results_by_category[category_prefix].append(result)
        
        # Brief pause between tests
        time.sleep(0.2)
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print("\n📊 RESULTS BY CATEGORY:")
    print("-"*40)
    
    for category, results in sorted(results_by_category.items()):
        print(f"\n{category} ({len(results)} tests):")
        for r in results[:3]:  # Show first 3 examples
            input_preview = r["input"][:50] + "..." if len(r["input"]) > 50 else r["input"]
            response_preview = r["response"][:50] + "..." if len(r["response"]) > 50 else r["response"]
            print(f"  Q: {input_preview}")
            print(f"  A: {response_preview}")
    
    print("\n" + "="*60)
    print("CAPABILITY ASSESSMENT")
    print("="*60)
    
    print("\n✅ STRENGTHS (What the model learned):")
    print("  • Conversation format (User/Assistant structure)")
    print("  • Basic response generation")
    print("  • Some code formatting patterns")
    
    print("\n⚠️ WEAKNESSES (Areas needing improvement):")
    print("  • Factual accuracy")
    print("  • Mathematical computation")
    print("  • Logical reasoning")
    print("  • Instruction precision")
    
    print("\n📈 INSIGHTS:")
    print("  • Lower temperature (0.1) reveals deterministic patterns")
    print("  • Model shows format learning > semantic understanding")
    print("  • 112M parameters + 3hr training = pattern matching, not reasoning")
    print("  • GSM8K format learned but computation fails")
    
    print("\n💡 RECOMMENDATIONS:")
    print("  • More pretraining would improve base knowledge")
    print("  • Focused math dataset for arithmetic capabilities")
    print("  • Instruction diversity needed for better following")
    print("  • Consider knowledge distillation from larger models")
    
    print("="*60)

def interactive_chat(model, tokenizer):
    """Interactive chat mode."""
    print("\n" + "="*60)
    print("INTERACTIVE CHAT MODE")
    print("Type 'quit' to exit, 'help' for options")
    print("="*60)
    
    settings = {
        'temperature': 0.1,
        'top_k': 40,
        'top_p': 0.85,
        'repetition_penalty': 1.1,
        'max_tokens': 550
    }
    
    print("\n[INFO] This is an instruction-tuned model.")
    print("   Just type your questions or requests naturally!")
    
    while True:
        user_input = input("\nYou: ").strip()
        
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
                print(f"[OK] Temperature set to {settings['temperature']}")
            except:
                print("[ERROR] Invalid temperature")
            continue
        elif user_input.startswith('max='):
            try:
                settings['max_tokens'] = int(user_input.split('=')[1])
                print(f"[OK] Max tokens set to {settings['max_tokens']}")
            except:
                print("[ERROR] Invalid max tokens")
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
            print(f"\nAssistant:\n{response}")

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
        print(f"[ERROR] Checkpoint '{args.checkpoint}' not found!")
        print(f"   Looking for: {args.checkpoint}")
        print(f"   Make sure SFT training has completed and saved the model.")
        return
    
    # Load model and tokenizer
    model, config = load_sft_model(args.checkpoint)
    tokenizer = get_tokenizer()
    
    print("\n[READY] SFT Model Ready!")
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
            print("[ERROR] Please provide an instruction with --input")
            return
        response = generate_response(
            model, tokenizer, args.input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"\n[RESPONSE]\n{response}")

if __name__ == "__main__":
    main()