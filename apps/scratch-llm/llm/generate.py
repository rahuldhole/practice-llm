import torch
from llm.tokenizer import SimpleTokenizer
from llm.model import ScratchLLM
import json

def load_tokenizer(vocab_path):
    with open(vocab_path, "r") as f:
        word2id = json.load(f)
    
    tokenizer = SimpleTokenizer()
    tokenizer.word2id = word2id
    tokenizer.id2word = {int(v): k for k, v in word2id.items()}
    tokenizer.vocab_size = len(word2id)
    return tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=10):
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Convert to tensor and add batch dimension
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            
            # Forward pass
            logits = model(x) # (1, seq_len, vocab_size)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]
            
            # Greedy decoding (argmax)
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Append to sequence
            tokens.append(next_token_id)
            
            # Stop if we generate a [PAD] or something else (optional)
            # if next_token_id == tokenizer.pad_token_id: break
            
    # Decode and return
    return tokenizer.decode(tokens)
