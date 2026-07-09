# Blueprint: Byte-Level Pre-training (Tokenizer-Free)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Standard autoregressive Cross-Entropy Loss computed directly over UTF-8 byte streams ($0 - 255$) instead of subword token IDs.
  - Alleviates tokenization bias, vocabulary fragmentation, and out-of-vocabulary (OOV) issues.
- **FLOPs Scaling Formula:**
  - Byte-level models require approximately $4\times$ more sequence length to represent equivalent semantic content as a standard tokenizer.
  - $\text{FLOPs} \approx 6 \times N \times (D \times 4)$ where $N$ is parameters and $D$ is the standard token-equivalent data volume.
- **Precision Profile:**
  - **Shared Trunk Layers:** BF16.
  - **Embedding & Output Projection Head:** FP32 (vocabulary size is fixed at exactly 256 + special control characters, making output projections cheap).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM).
- **Provider:** RunPod or Vast.ai.
- **Execution Duration Limit:** < 10 hours.
- **Target Token/Batch Scale:** 100M-500M bytes. Batch size 16 at 8K context.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 (SXM5, 80GB) node.
- **Interconnect Requirements:** NVLink (intra-node) for high-frequency gradient synchronization.
- **Persistent Storage Topography:** High-performance local NVMe scratch disk to handle massive raw byte file stream reads.

## 3. Data Topography
- **Token Window Length:** 32,768 bytes (roughly equivalent to 8,000 subword tokens).
- **Preprocessing Requirements:** Direct conversion of raw texts to UTF-8 binary encoding. No BPE, WordPiece, or SentencePiece step.
- **Tokenizer Handling:** No tokenizer is used. Input dimensions are fixed at 256.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Training loop snippet for character/byte-level autoregressive training.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class ByteDataset(Dataset):
    def __init__(self, filepath, seq_len=4096):
        with open(filepath, 'rb') as f:
            self.data = f.read()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        # Read seq_len bytes as input, and offset by 1 as target
        chunk = self.data[idx:idx + self.seq_len + 1]
        tensor = torch.tensor(list(chunk), dtype=torch.long)
        return tensor[:-1], tensor[1:]

# Simple character/byte transformer head training CLI entrypoint
if __name__ == "__main__":
    dataset = ByteDataset("highly_specialized_logs.txt", seq_len=4096)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Vocabulary size is exactly 256 bytes
    model = nn.Sequential(
        nn.Embedding(256, 512),
        # Standard transformer blocks here...
        nn.Linear(512, 256)
    ).cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits.view(-1, 256), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
        break
```

## 5. Failure Modes & Recovery
- **Slow Semantic Convergence:**
  - *Indicator:* Loss decreases, but model struggles to output coherent words or phrases, instead learning character patterns very slowly.
  - *Mitigation:* Employ Hierarchical Transformer architectures (e.g., MegaByte) where a slow global transformer routes representations to a fast local byte transformer.
- **Context Window Bottleneck:**
  - *Indicator:* Out of memory (OOM) due to quadratic attention scaling across very long byte sequences.
  - *Mitigation:* Implement FlashAttention-2 or linear attention kernels to scale to 32k+ contexts efficiently.
