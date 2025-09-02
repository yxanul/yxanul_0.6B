from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
