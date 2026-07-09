import torch
import torch.nn as nn
import torch.optim as optim
import random
from basics.tokenizer import CharTokenizer

class RNNCell(nn.Module):
    """
    A single Recurrent Neural Network cell, implemented from scratch using PyTorch linear weights.
    h_t = tanh(x_t * W_xh^T + h_{t-1} * W_hh^T + b_h)
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Linear layer for input-to-hidden
        self.i2h = nn.Linear(input_size, hidden_size)
        # Linear layer for hidden-to-hidden
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False) # bias is kept in i2h

    def forward(self, x, h):
        # x: shape (batch, input_size)
        # h: shape (batch, hidden_size)
        return torch.tanh(self.i2h(x) + self.h2h(h))

class RNNLanguageModel(nn.Module):
    """
    An RNN character-level language model.
    Encodes characters, processes sequences step-by-step with an RNNCell, and projects to vocab logits.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 1. Embedding layer: turns token IDs into vectors
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Our custom RNN cell
        self.rnn = RNNCell(embedding_dim, hidden_size)
        
        # 3. Output projection: maps hidden states back to vocabulary logits
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: shape (batch, seq_len)
        # targets: shape (batch, seq_len)
        batch_size, seq_len = idx.shape
        
        # Initialize hidden state to zeros
        h = torch.zeros(batch_size, self.hidden_size, device=idx.device)
        
        # Embed the tokens: shape (batch, seq_len, embedding_dim)
        embeddings = self.token_embedding(idx)
        
        # Step through the sequence character by character
        logits_list = []
        for t in range(seq_len):
            x_t = embeddings[:, t, :] # shape (batch, embedding_dim)
            h = self.rnn(x_t, h)      # shape (batch, hidden_size)
            logits_t = self.lm_head(h) # shape (batch, vocab_size)
            logits_list.append(logits_t)
            
        # Stack logits: shape (batch, seq_len, vocab_size)
        logits = torch.stack(logits_list, dim=1)
        
        loss = None
        if targets is not None:
            # Flatten logits and targets to compute cross entropy loss
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens):
        # start_tokens: shape (1, seq_len)
        idx = start_tokens
        batch_size = idx.shape[0]
        
        # Run the prompt through the model to compute the final hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=idx.device)
        embeddings = self.token_embedding(idx)
        
        # Step through prompt tokens
        for t in range(idx.shape[1]):
            x_t = embeddings[:, t, :]
            h = self.rnn(x_t, h)
            
        # Now generate the next tokens autoregressively
        generated = []
        # Current token is the last token in the prompt
        curr_idx = idx[:, -1:]
        
        for _ in range(max_new_tokens):
            x_t = self.token_embedding(curr_idx).squeeze(1) # shape (batch, embedding_dim)
            h = self.rnn(x_t, h)
            logits = self.lm_head(h) # shape (batch, vocab_size)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample next token
            next_idx = torch.multinomial(probs, num_samples=1)
            generated.append(next_idx)
            curr_idx = next_idx
            
        return torch.cat(generated, dim=1)


if __name__ == "__main__":
    print("--- RNN Character Language Model Demo ---")
    torch.manual_seed(42)
    random.seed(42)
    
    # 1. Dataset
    corpus = "hello world! hello from scratch transformer!"
    tokenizer = CharTokenizer(corpus)
    print(f"Dataset Vocab Size: {tokenizer.vocab_size}")
    
    # Let's create input/target pairs
    # E.g. input: "hello worl" -> target: "ello world"
    encoded_corpus = tokenizer.encode(corpus)
    
    # We will generate training batches
    seq_len = 8
    inputs = []
    targets = []
    for i in range(len(encoded_corpus) - seq_len):
        inputs.append(encoded_corpus[i : i + seq_len])
        targets.append(encoded_corpus[i + 1 : i + seq_len + 1])
        
    X = torch.tensor(inputs)  # (N, seq_len)
    Y = torch.tensor(targets) # (N, seq_len)
    
    # 2. Initialize RNN Language Model
    model = RNNLanguageModel(tokenizer.vocab_size, embedding_dim=16, hidden_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Simple Training Loop
    print("Training character-level RNN prediction model...")
    for epoch in range(120):
        # Forward pass
        logits, loss = model(X, Y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
            
    # 4. Generate text from prompt
    prompt = "hello "
    prompt_tensor = torch.tensor([tokenizer.encode(prompt)]) # shape (1, 6)
    generated_ids = model.generate(prompt_tensor, max_new_tokens=20)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated text: '{prompt + generated_text}'")
