# Blueprint: Reward Model Training

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Train a scalar score regression head $V_\phi$ on top of a base transformer model to evaluate quality of generated responses.
  - Aligns the reward predictions with human preferences by optimizing the Bradley-Terry preference loss:
    $$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( V_\phi(x, y_w) - V_\phi(x, y_l) \right) \right]$$
    where $y_w$ is the chosen (preferred) response and $y_l$ is the rejected response.
- **FLOPs Scaling Formula:**
  - Requires forward passes for both chosen and rejected sequences.
  - $\text{FLOPs} \approx 6 \times N \times D_{\text{pairs}}$ where $D_{\text{pairs}}$ is the total number of tokens inside both chosen and rejected sequences.
- **Precision Profile:**
  - **Base model backbone:** BF16.
  - **Scalar head layer:** FP32.
  - **Loss computation:** FP32 (to prevent overflow/underflow in log-sigmoid calculation).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM).
- **Provider:** RunPod.
- **Execution Duration Limit:** < 6 hours.
- **Target Token/Batch Scale:** 20,000 preference pairs (approx. 40M tokens total).

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 SXM5 node.
- **Interconnect Requirements:** NVLink (intra-node) for high-speed weights sync.
- **Persistent Storage Topography:** High-speed NVMe scratch storage.

## 3. Data Topography
- **Token Window Length:** 4,096 tokens (must cover prompt + response lengths).
- **Preprocessing Requirements:** Input formatting: concatenating prompt + choice vs. prompt + rejection. Data must be balanced to ensure safety chosen labels do not override capability scores completely.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Script using TRL (Transformer Reinforcement Learning) to train a Reward Model.

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

def train_reward_model(model_name, dataset_path):
    # Load model with a single classification head/output value
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1, 
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Load pair dataset (chosen vs rejected)
    # Target dataset must contain columns: "input_ids_chosen", "attention_mask_chosen", 
    # "input_ids_rejected", and "attention_mask_rejected".
    
    training_args = RewardConfig(
        output_dir="./reward_model_checkpoints",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        logging_steps=10,
        bf16=True,
        evaluation_strategy="steps",
        eval_steps=100,
    )
    
    # Instantiate trainer (requires dummy dataset for execution)
    # trainer = RewardTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     train_dataset=my_dataset
    # )
    # trainer.train()
    print("Reward Model trainer initialization verified.")

if __name__ == "__main__":
    train_reward_model("Qwen/Qwen2.5-3B", "dummy_dataset")
```

## 5. Failure Modes & Recovery
- **Reward Overoptimization / Score Exploitation:**
  - *Indicator:* During PPO reinforcement learning, the generator model outputs short, repetitive, or nonsensical tokens that get high scores from the reward model.
  - *Mitigation:* Apply KL-divergence penalties to keep the active policy close to the initial base policy. Inject a penalty for generation length outliers.
- **Class Imbalance / Uniform Predictions:**
  - *Indicator:* The reward model outputs a static score (e.g. constant 0.5) regardless of input quality.
  - *Mitigation:* Ensure diversity in preference training pairs (mix easy choices with fine-grained critiques). Verify scalar head initialization is non-zero.
