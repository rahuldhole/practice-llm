import torch
from llm.tokenizer import SimpleTokenizer
from llm.model import ScratchLLM

def load_tokenizer(vocab_path):
    if vocab_path.endswith("vocab.json"):
        vocab_path = vocab_path.replace("vocab.json", "tokenizer.json")
    return SimpleTokenizer.load(vocab_path)

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=10, device="cpu"):
    model.eval()
    device = torch.device(device)
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.pad_token_id]
        
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Process the full sequence each time (no KV cache)
            logits = model(x)
            
            # Select the last token's logits and scale by temperature
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
            
            # Apply Top-K filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
                
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Append token to active sequences
            tokens.append(next_token_id)
            x = torch.cat([x, torch.tensor([[next_token_id]], device=device)], dim=1)
            
            # Stop if we hit a pad token
            if next_token_id == tokenizer.pad_token_id:
                break
                
    return tokenizer.decode(tokens)
