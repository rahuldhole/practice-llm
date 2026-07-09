import torch
from llm.tokenizer import SimpleTokenizer
from llm.model import ScratchLLM

def load_tokenizer(vocab_path):
    if vocab_path.endswith("vocab.json"):
        vocab_path = vocab_path.replace("vocab.json", "tokenizer.json")
    return SimpleTokenizer.load(vocab_path)

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=10, device="cpu"):
    from llm.model import KVCache
    model.eval()
    device = torch.device(device)
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.pad_token_id]
        
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    kv_cache = KVCache()
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            if i == 0:
                # Step 1: Process the full prompt to populate the KV cache
                logits = model(x, start_pos=0, kv_cache=kv_cache)
            else:
                # Step 2+: Process only the single most recently generated token
                logits = model(x[:, -1:], start_pos=x.shape[1] - 1, kv_cache=kv_cache)
            
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
