# 🏗️ Tutorial 07: Transformers & Tiny GPT

By stacking multi-head attention, normalization layers, residual connections, and position-wise feed-forward networks, we build a **Decoder-only Transformer** (like GPT-4). This architecture acts as the backbone of modern Generative AI.

---

## 1. Positional Encoding

Attention mechanisms process all tokens in parallel, meaning attention is **permutation-invariant**. Without positional information, a Transformer would treat the sentences `"cat eats fish"` and `"fish eats cat"` exactly the same!

To fix this, we assign each position (index $0, 1, 2, \dots$) its own learned vector, which we add directly to the token embeddings:
$$\mathbf{X}_{\text{final}} = \mathbf{X}_{\text{token}} + \mathbf{X}_{\text{position}}$$

---

## 2. Pre-LN Transformer Block

A Transformer is built by stacking identical layers called **Transformer Blocks**. In modern models (Pre-LN architecture), each block performs two main sub-steps with residual connections and layer normalization:

```text
Input (x) 
  │
  ├───> LayerNorm ───> Multi-Head Attention ───> (Add) ───> Residual Stream (x_1)
  │                                               ▲
  └───────────────────────────────────────────────┘
  │
  ├───> LayerNorm ───> Feed-Forward Network ───> (Add) ───> Output Stream (x_2)
  │                                               ▲
  └───────────────────────────────────────────────┘
```

### A. Residual Connections (Skip Connections)
Instead of forcing gradients to propagate through complex non-linear functions during backpropagation, residual connections add the input directly to the block output:
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{SubLayer}(\mathbf{x}_l)$$

This acts as a "gradient highway," enabling us to train hundreds of layers without gradients vanishing.

### B. Layer Normalization (LayerNorm)
LayerNorm normalizes the features of each token independently:
$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
where $\mu$ is the mean of the token's features, and $\sigma^2$ is its variance. This stabilizes the scale of activations across deep layers.

*Code reference*: [`Block` in gpt.py](file:///home/ubuntu/playground/practice-llm/apps/basics/src/gpt.py#L20-L38)

---

## 3. Position-Wise Feed-Forward Network (FFN)

After attention extracts contextual information from the sequence, a Feed-Forward Network processes each token independently:
$$\text{FFN}(x) = \text{Activation}(x \mathbf{W}_1 + b_1) \mathbf{W}_2 + b_2$$

The FFN typically projects the embedding dimension to $4 \times$ its size before projecting it back. This expansion acts as a database/storage space where the model encodes factual knowledge.

*Code reference*: [`FeedForward` in gpt.py](file:///home/ubuntu/playground/practice-llm/apps/basics/src/gpt.py#L7-L18)

---

## 4. The Tiny GPT Model Architecture

Putting it all together, our `TinyGPT` decoder model:
1. Embeds tokens and adds position embeddings.
2. Passes representations through a sequential stack of `Block`s.
3. Normalizes outputs using a final `LayerNorm`.
4. Projects representations to vocabulary logits to predict the next token.

*Code reference*: [`TinyGPT` in gpt.py](file:///home/ubuntu/playground/practice-llm/apps/basics/src/gpt.py#L40-L100)

---

## 💡 Practical Challenge
Run `task run -- src/gpt.py`. Observe the training loss curves. Try changing the context window size (`block_size`) in `gpt.py` and see how it affects training memory and text generation coherence.
