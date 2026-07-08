import torch
from torch.utils.data import Dataset, DataLoader
from llm.tokenizer import SimpleTokenizer

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence
        x = self.tokens[idx : idx + self.seq_length]
        # Target sequence (shifted by one)
        y = self.tokens[idx + 1 : idx + self.seq_length + 1]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def get_dataloader(file_path, seq_length, batch_size, vocab_size=2000):
    with open(file_path, "r") as f:
        text = f.read()
    
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.fit(file_path)
    
    dataset = TextDataset(text, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader, tokenizer

if __name__ == "__main__":
    dataloader, tokenizer = get_dataloader("data/dataset.txt", seq_length=4, batch_size=2)
    print("Vocab size:", tokenizer.vocab_size)
    for x, y in dataloader:
        print("x:", x)
        print("y:", y)
        break
