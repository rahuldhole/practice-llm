import pytest
import torch
from torch.utils.data import DataLoader
from pytorch_patterns.custom_dataset import TextSequenceDataset, PadCollate

def test_text_sequence_dataset():
    texts = ["cat", "elephant", "dog"]
    dataset = TextSequenceDataset(texts)
    
    # Check length
    assert len(dataset) == 3
    
    # Check vocabulary mapping
    assert dataset.pad_idx in dataset.idx_to_char
    assert dataset.char_to_idx[dataset.pad_char] == dataset.pad_idx
    
    # Check getitem
    seq, seq_len = dataset[0]
    assert isinstance(seq, torch.Tensor)
    assert seq.dtype == torch.long
    assert seq_len == 3
    
    # Decode verification
    decoded = dataset.decode(seq.tolist())
    assert decoded == "cat"


def test_pad_collate():
    texts = ["a", "abc", "ab"]
    dataset = TextSequenceDataset(texts)
    
    collate_fn = PadCollate(pad_idx=dataset.pad_idx)
    
    # Simulate list of samples returned by __getitem__
    samples = [dataset[i] for i in range(len(dataset))]
    
    padded_batch, lengths = collate_fn(samples)
    
    # Since batch is sorted reverse, max length should be first: "abc" (length 3)
    assert padded_batch.shape == (3, 3)
    assert lengths.tolist() == [3, 2, 1]
    
    # Verify padding index value
    # "a" (length 1) padded with 2 values of pad_idx at the end
    assert padded_batch[2, 0] != dataset.pad_idx
    assert padded_batch[2, 1] == dataset.pad_idx
    assert padded_batch[2, 2] == dataset.pad_idx
