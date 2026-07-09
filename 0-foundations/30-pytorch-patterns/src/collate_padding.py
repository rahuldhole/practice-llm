import torch

class PadCollate:
    """
    A custom collate helper that pads shorter sequences in a batch
    with zero tokens so they all match the length of the longest sequence.
    """
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # 1. Sort the batch by sequence length from longest to shortest
        batch = sorted(batch, key=lambda x: x[1], reverse=True)
        sequences, lengths = zip(*batch)
        
        # 2. Find the longest sequence length in this specific batch
        max_len = max(lengths)
        
        # 3. Pad each sequence to match max_len
        padded_sequences = []
        for seq in sequences:
            pad_needed = max_len - len(seq)
            if pad_needed > 0:
                # pad syntax: (pad_left, pad_right)
                padded_seq = torch.nn.functional.pad(seq, (0, pad_needed), value=self.pad_idx)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
            
        # 4. Stack all padded sequences into a single 2D batch tensor
        padded_batch = torch.stack(padded_sequences, dim=0)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        return padded_batch, lengths_tensor


if __name__ == "__main__":
    print("--- Collate Padding Demo ---")
    # Simulate a batch of 2 sequences of lengths 3 and 1
    sample_batch = [
        (torch.tensor([1, 2, 3]), 3),
        (torch.tensor([5]), 1)
    ]
    
    collator = PadCollate(pad_idx=0)
    padded_batch, lengths = collator(sample_batch)
    
    print("Padded Batch Tensor:\n", padded_batch)
    print("Sequence Lengths:\n", lengths)
