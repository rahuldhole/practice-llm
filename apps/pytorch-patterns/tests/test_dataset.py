import pytest
import torch
from pytorch_patterns.dataset_text import TextSequenceDataset
from pytorch_patterns.collate_padding import PadCollate

def test_text_sequence_dataset():
    texts = ["cat", "elephant"]
    dataset = TextSequenceDataset(texts)
    
    assert len(dataset) == 2
    seq, seq_len = dataset[0]
    assert isinstance(seq, torch.Tensor)
    assert seq.dtype == torch.long
    assert seq_len == 3
    
    assert dataset.decode(seq.tolist()) == "cat"


def test_pad_collate():
    texts = ["a", "abc"]
    dataset = TextSequenceDataset(texts)
    collator = PadCollate(pad_idx=dataset.pad_idx)
    
    samples = [dataset[0], dataset[1]]
    padded_batch, lengths = collator(samples)
    
    # "abc" is length 3, "a" is length 1 (padded with 2 zeros at the end)
    assert padded_batch.shape == (2, 3)
    assert lengths.tolist() == [3, 1]
    assert padded_batch[1, 1] == dataset.pad_idx
    assert padded_batch[1, 2] == dataset.pad_idx
