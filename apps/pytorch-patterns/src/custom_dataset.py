import torch
from torch.utils.data import Dataset, DataLoader

class TextSequenceDataset(Dataset):
    """
    A custom PyTorch Dataset that loads raw text sequences, maps characters to
    integer tokens, and returns PyTorch tensors representing character index lists.
    
    This illustrates the classic pattern of implementing __len__ and __getitem__
    to prepare raw sequence data for neural network consumption.
    """
    def __init__(self, texts, pad_char="<PAD>"):
        """
        Args:
            texts (list of str): List of input sentences.
            pad_char (str): Special character to reserve for padding.
        """
        self.texts = texts
        self.pad_char = pad_char
        
        # Build a character vocabulary
        unique_chars = sorted(list(set("".join(texts))))
        
        # Ensure pad character is in the vocabulary
        if pad_char not in unique_chars:
            unique_chars = [pad_char] + unique_chars
            
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.pad_idx = self.char_to_idx[pad_char]

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves a single text sample, tokenizes it into character IDs,
        and returns it as a PyTorch tensor.
        
        Returns:
            torch.Tensor: LongTensor containing token index sequence.
            int: The original length of the sequence.
        """
        text = self.texts[idx]
        tokens = [self.char_to_idx[c] for c in text]
        return torch.tensor(tokens, dtype=torch.long), len(tokens)

    def decode(self, token_ids):
        """
        Decodes a list of token IDs back into a string representation.
        """
        return "".join([self.idx_to_char.get(int(tid), "?") for tid in token_ids])


class PadCollate:
    """
    A custom collate function wrapper (implemented as a callable class)
    to dynamically pad sequences of varying lengths within a mini-batch.
    
    In deep learning, batches of inputs must be stacked into a single tensor,
    requiring uniform dimensions. Padding dynamically per-batch prevents
    unnecessary overhead of padding all sequences to a global maximum length.
    """
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        Collates a list of (sequence_tensor, length) tuples into a batch.
        
        Args:
            batch (list of tuple): A list of samples returned by Dataset.__getitem__.
            
        Returns:
            torch.Tensor: Padded sequence tensor of shape (batch_size, max_len).
            torch.Tensor: Tensor containing original lengths of the sequences.
        """
        # Sort batch by sequence length (optional, but often useful for RNN packing)
        batch = sorted(batch, key=lambda x: x[1], reverse=True)
        sequences, lengths = zip(*batch)
        
        # Find maximum length in this specific batch
        max_len = max(lengths)
        
        # Pad each sequence to the maximum batch length
        padded_sequences = []
        for seq in sequences:
            pad_size = max_len - len(seq)
            if pad_size > 0:
                # pad syntax: (pad_left, pad_right)
                padded_seq = torch.nn.functional.pad(seq, (0, pad_size), value=self.pad_idx)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
            
        # Stack all padded sequences into a single batch tensor
        padded_batch = torch.stack(padded_sequences, dim=0)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        return padded_batch, lengths_tensor


if __name__ == "__main__":
    print("--- PyTorch Custom Dataset & DataLoader Demo ---")
    
    # 1. Define synthetic variable-length texts
    sample_texts = [
        "PyTorch",
        "Deep Learning",
        "Transformer Networks",
        "Attention Is All You Need",
        "SGD",
    ]
    
    print("Input sequences:")
    for text in sample_texts:
        print(f" - '{text}' (Length: {len(text)})")
        
    # 2. Instantiate Dataset
    dataset = TextSequenceDataset(sample_texts)
    print(f"\nVocabulary size: {len(dataset.char_to_idx)}")
    print(f"Padding index: {dataset.pad_idx}")
    
    # 3. Create DataLoader using our PadCollate class
    # We use a batch size of 2 to observe dynamic padding behavior per batch.
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=PadCollate(pad_idx=dataset.pad_idx)
    )
    
    print("\nIterating over DataLoader batches:")
    for i, (padded_seqs, lengths) in enumerate(loader):
        print(f"\n--- Batch {i+1} ---")
        print(f"Padded Sequence Tensor Shape: {padded_seqs.shape}")
        print(f"Original Sequence Lengths: {lengths.tolist()}")
        print("Padded Batch Tensor:")
        print(padded_seqs)
        
        print("Decoded batches (padding marked as <PAD>):")
        for seq in padded_seqs:
            decoded = dataset.decode(seq.tolist())
            print(f" - '{decoded}'")
