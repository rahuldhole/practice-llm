# Blueprint: Iterative / Online Preference Optimization

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Prevents distribution drift by querying preferences on-policy.
  - Unlike standard offline DPO (which uses a fixed, historical chosen/rejected dataset), Online DPO queries the active student policy $\pi_\theta$ to generate new rollouts $y_1, y_2$ at each training step.
  - A reward model (or verifier) $R$ scores these rollouts. The highest scoring rollout becomes the chosen sample $y_w$, and the lowest scoring becomes the rejected sample $y_l$.
  - The DPO loss is computed using these dynamically generated pairs relative to the reference model $\pi_{\text{ref}}$:
    $$\mathcal{L}_{\text{Online-DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)$$
- **FLOPs Scaling Formula:**
  - Generates rollouts at each iteration, requiring a mix of inference forward/backward passes.
  - $\text{FLOPs} \approx I \times (2 \times N_{\text{inference}} + 9 \times N_{\text{train}}) \times D_{\text{batch}}$ where $I$ is number of online iterations.
- **Precision Profile:**
  - **Active Policy & Reference Backbones:** BF16.
  - **Reward Model / Verifier:** BF16/FP16.
  - **Loss Logs:** FP32.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM) or 1x A100 (80GB).
- **Provider:** RunPod.
- **Execution Duration Limit:** < 8 hours.
- **Target Token/Batch Scale:** 3 iterations, 5,000 prompts per iteration.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 SXM5 node (to fit active policy, reference model, and reward model in VRAM simultaneously).
- **Interconnect Requirements:** NVLink (intra-node) for high-speed weights sync.
- **Persistent Storage Topography:** High speed local scratch NVMe.

## 3. Data Topography
- **Token Window Length:** 4,096 tokens.
- **Preprocessing Requirements:** General instructions or reasoning prompts. Set a temperature parameter (e.g. 0.8) during rollout generation to ensure adequate output diversity.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Conceptual training script for Online/Iterative DPO.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

def run_online_dpo_iteration(model_path, ref_model_path, reward_model, prompts):
    # 1. Load active policy and reference models
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    online_pairs = []
    
    # 2. On-policy Rollout & Scoring Loop
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate 2 distinct samples
        outputs = model.generate(
            **inputs,
            num_return_sequences=2,
            do_sample=True,
            temperature=0.8,
            max_new_tokens=256
        )
        
        cand1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cand2 = tokenizer.decode(outputs[1], skip_special_tokens=True)
        
        # Score candidates using Reward Model (mocked)
        score1 = reward_model(prompt, cand1)
        score2 = reward_model(prompt, cand2)
        
        if score1 > score2:
            online_pairs.append({"prompt": prompt, "chosen": cand1, "rejected": cand2})
        else:
            online_pairs.append({"prompt": prompt, "chosen": cand2, "rejected": cand1})
            
    # 3. Apply standard DPO optimization step using TRL DPOTrainer
    # training_args = DPOConfig(output_dir="./online_dpo_checkpoints", bf16=True)
    # trainer = DPOTrainer(model=model, ref_model=ref_model, args=training_args, train_dataset=online_pairs)
    # trainer.train()
    
    print(f"Online DPO iteration completed with {len(online_pairs)} samples.")

if __name__ == "__main__":
    def dummy_reward(p, c):
        return len(c) # Mock reward: longer is better
    run_online_dpo_iteration("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B", dummy_reward, ["Explain quantum physics."])
```

## 5. Failure Modes & Recovery
- **Reward Hacker Output Spikes:**
  - *Indicator:* The student model learns to generate short phrases full of punctuation or emotional appeals that trigger high reward scores, losing actual instructions response capability.
  - *Mitigation:* Apply strict KL divergence regularization ($\beta = 0.05 - 0.1$) to prevent the active policy from shifting too far from the reference model.
- **Reference Model Divergence:**
  - *Indicator:* Training losses stay flat or become NaN as active model distributions diverge completely.
  - *Mitigation:* Decrease learning rate (e.g. to 5e-6) and update the reference model checkpoint at the start of each iteration.
