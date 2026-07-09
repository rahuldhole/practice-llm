import torch
from torch.utils.data import Dataset

class TextSequenceDataset(Dataset):
    """
    A simple dataset that reads text sentences and turns them into numbers.
    Like a dictionary translating letters (chars) into page numbers.
    """
    def __init__(self, texts, pad_char="<PAD>"):
        self.texts = texts
        self.pad_char = pad_char
        
        # 1. Gather all unique characters across all sentences
        unique_chars = sorted(list(set("".join(texts))))
        
        # 2. Put padding character first so its ID is 0
        if pad_char not in unique_chars:
            unique_chars = [pad_char] + unique_chars
            
        # 3. Create bidirectional maps: character <-> integer ID
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.pad_idx = self.char_to_idx[pad_char]

    def __len__(self):
        # Tells PyTorch how many sentences we have
        return len(self.texts)

    def __getitem__(self, idx):
        # Get one sentence by its position index
        text = self.texts[idx]
        
        # Translate each character in the sentence to its number ID
        tokens = [self.char_to_idx[char] for char in text]
        
        # Return as a PyTorch sequence tensor and its original length
        return torch.tensor(tokens, dtype=torch.long), len(tokens)

    def decode(self, token_ids):
        # Translate integer IDs back to characters to read the text
        return "".join([self.idx_to_char.get(int(tid), "?") for tid in token_ids])


if __name__ == "__main__":
    print("--- Toy Dataset Demo ---")
    sentences = ["cat", "elephant"]
    dataset = TextSequenceDataset(sentences)
    
    print(f"Vocab character IDs: {dataset.char_to_idx}")
    print(f"Padding index: {dataset.pad_idx}")
    
    # Grab the first item
    seq, seq_len = dataset[0]
    print(f"First text: '{sentences[0]}' -> Tensor: {seq}, Length: {seq_len}")
