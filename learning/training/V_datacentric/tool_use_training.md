# Blueprint: Tool-Use / Function-Calling SFT

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Train model to interface with external execution environments (calculators, databases, APIs) by emitting special markup blocks and parsing output logs.
  - Standard SFT cross-entropy loss is applied, but critically, we **mask out the loss calculation** for the tokens returned by the tool response. The model must only learn to *emit* the call and *synthesize* the final response, not predict the output of the external tools:
    $$\mathcal{L} = -\sum_{t \in \mathcal{T}_{\text{model\_tokens}}} \log P(y_t \mid y_{<t}, x)$$
    where $\mathcal{T}_{\text{model\_tokens}}$ excludes the token span representing the execution return output.
- **FLOPs Scaling Formula:**
  - Follows standard SFT compute requirements:
    $$\text{FLOPs} \approx 6 \times N \times D_{\text{trajectories}}$$
- **Precision Profile:**
  - **Backbone & Projections:** BF16/FP16.
  - **Loss computation:** FP32.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM).
- **Provider:** RunPod.
- **Execution Duration Limit:** < 4 hours.
- **Target Token/Batch Scale:** 20,000 multi-turn trajectories.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 SXM5 node.
- **Interconnect Requirements:** NVLink (intra-node).
- **Persistent Storage Topography:** High speed local scratch NVMe storage.

## 3. Data Topography
- **Token Window Length:** 8,192 tokens.
- **Preprocessing Requirements:** Data formatted with distinct tags, e.g.:
  `[Prompt] <thought>...</thought> <call:calculator>{"args": [2, 3]}</call:calculator> [Tool Response] <response>5</response> <thought>...</thought> [Final Answer]`.
  The token span between `<response>` and `</response>` must have its target labels set to `-100` in PyTorch's `CrossEntropyLoss` to bypass gradient updates.
- **Tokenizer Handling:** Register special tags (`<call:`, `</call:`, `<response>`, `</response>`) as new special tokens in the tokenizer to prevent fragmentation.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Preprocessing script illustrating target label masking for tool-use data.

import torch

def mask_tool_responses_in_loss(input_ids, tokenizer, loss_mask_value=-100):
    # input_ids: tensor of token IDs representing prompt + call + response + final
    # Label masking: set parts we do NOT want the model to learn to predict to -100
    labels = input_ids.clone()
    
    # Identify token IDs for tags
    response_start_token = tokenizer.encode("<response>", add_special_tokens=False)[0]
    response_end_token = tokenizer.encode("</response>", add_special_tokens=False)[0]
    
    in_response_block = False
    for i in range(len(input_ids)):
        token = input_ids[i].item()
        
        if token == response_start_token:
            in_response_block = True
            labels[i] = loss_mask_value # Mask the tag itself
            continue
        elif token == response_end_token:
            in_response_block = False
            labels[i] = loss_mask_value # Mask the tag itself
            continue
            
        if in_response_block:
            # Mask this index so the model does not try to predict the external return value
            labels[i] = loss_mask_value
            
    return labels

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<response>", "</response>"]})
    
    dummy_text = "Compute 2+2. <call:calc>2+2</call:calc> <response> 4 </response> The answer is 4."
    tokens = torch.tensor(tokenizer.encode(dummy_text))
    labels = mask_tool_responses_in_loss(tokens, tokenizer)
    
    # Print out aligned tokens and labels to verify masking
    for t, l in zip(tokens, labels):
        print(f"Token: {tokenizer.decode([t.item()]):<12} Label ID: {l.item()}")
```

## 5. Failure Modes & Recovery
- **Syntactic Hallucinations (Malformed JSON/XML):**
  - *Indicator:* The model outputs malformed function parameters (e.g. missing commas or brackets) which causes execution parser crashes.
  - *Mitigation:* Enforce output structure during generation using guided decoding tools (e.g. Outlines, Instructor) or add schema-assertion few-shot examples in SFT.
- **Infinite Execution Loops:**
  - *Indicator:* The model repeats tool call patterns over and over, calling the same function indefinitely without outputting a final answer.
  - *Mitigation:* Enforce a strict max-step parameter (e.g. maximum 5 tool calls per loop) in your orchestration layer. Penalize repetitive actions during SFT.
