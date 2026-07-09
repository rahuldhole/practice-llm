# 🧬 Tutorial 01: Custom Datasets & DataLoaders

**TLDR:** Handling variable-length sequence data using custom Datasets and batch collators.

In deep learning, data pipeline setup is just as important as model architecture. PyTorch provides a highly extensible data API consisting of the `Dataset` and `DataLoader` classes.

For simple datasets (like static images or fixed-length tables), default loaders stack samples out of the box. However, for sequence tasks (like NLP or speech processing), sequence lengths vary. Stacking them directly into a batch tensor fails because tensors must be rectangular.

This tutorial covers the pattern to resolve this: creating a custom dataset to map tokens, and implementing a custom `collate_fn` to pad sequences dynamically per-batch.

---

## 1. Custom Dataset Protocol
A custom dataset inherits from `torch.utils.data.Dataset` and implements:
- `__init__(self)`: Prepares the input source (e.g., reads files or takes lists) and builds mappings (like character/word vocabularies).
- `__len__(self)`: Returns the total number of samples.
- `__getitem__(self, idx)`: Returns a single tokenized input sample and its length.

*Code reference*: [TextSequenceDataset in custom_dataset.py](../src/custom_dataset.py#L4-L44)

---

## 2. Dynamic Padding with a Custom Collator
The standard `DataLoader` aggregates samples using a default collation function (`default_collate`). If you try to batch sequences of length 3, 5, and 7, the loader raises a shape mismatch runtime error.

To solve this:
1. Provide a custom callable `collate_fn` parameter to the `DataLoader`.
2. Find the maximum sequence length in that specific mini-batch.
3. Pad the shorter sequences in that batch with a padding token to match the maximum length.
4. Stack them into a clean tensor of shape `(batch_size, max_len)`.

This pattern of dynamic padding (padding per batch rather than padding everything to a global maximum sequence length) reduces memory usage and training time.

*Code reference*: [PadCollate in custom_dataset.py](../src/custom_dataset.py#L47-L91)

---

## 💡 Practical Challenge
Open [custom_dataset.py](../src/custom_dataset.py) and run it with `task pytorch-patterns:run -- src/custom_dataset.py`. Try modifying the collate function to output a binary mask tensor `(batch_size, max_len)` indicating where actual tokens are (1) vs padding characters (0). This mask is essential when implementing causal and padding attention masks in Transformers!
