# Blueprint: Rejection Sampling Fine-Tuning (RFT)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Trains a model on its own verified correct generation paths to reinforce reasoning capabilities.
  - For each prompt $x_i$, generate $K$ candidate solutions $y_{i,1}, y_{i,2}, \dots, y_{i,K}$.
  - Evaluate candidates using a compiler, unit test verifier, or ground truth match function. Keep only the correct outputs.
  - Aligns the model via standard Supervised Fine-Tuning (SFT) cross-entropy loss restricted to the verified correct prompt-response paths:
    $$\mathcal{L}_{\text{RFT}} = -\sum_{(x, y) \in \mathcal{D}_{\text{correct}}} \log P(y \mid x)$$
- **FLOPs Scaling Formula:**
  - Generation phase: $K \times N \times D_{\text{prompt}}$ inference FLOPs.
  - Training phase: Standard SFT compute bounds on the filtered target corpus:
    $$\text{FLOPs} \approx 6 \times N \times D_{\text{correct}}$$
- **Precision Profile:**
  - **Inference Rollouts:** FP16/BF16 (to speed up batch generation).
  - **Fine-tuning:** BF16.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM).
- **Provider:** RunPod.
- **Execution Duration Limit:** < 12 hours.
- **Target Token/Batch Scale:** 10,000 prompts, generating $K=8$ completions each (80,000 total sequences).

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 SXM5 node.
- **Interconnect Requirements:** NVLink (to run fast parallel batched inference across nodes).
- **Persistent Storage Topography:** High speed local scratch storage to cache validation result files.

## 3. Data Topography
- **Token Window Length:** 8,192 tokens.
- **Preprocessing Requirements:** Math/Coding question prompts. Solutions must contain structured outputs (e.g. within XML tags `<answer>...</answer>`) for easy parsing.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Script demonstrating rollout generation and programmatic code verifier filtering.

import subprocess
import json

def run_code_verifier(code_snippet):
    # Programmatic verification of candidate solution (e.g. running a python interpreter test)
    try:
        result = subprocess.run(
            ["python3", "-c", code_snippet],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and "AssertPassed" in result.stdout:
            return True
    except subprocess.TimeoutExpired:
        pass
    return False

def build_rejection_sampling_dataset(prompts, model_pipeline, K=8):
    dataset = []
    for prompt in prompts:
        # Generate K candidates
        candidates = model_pipeline.generate(prompt, num_return_sequences=K, max_length=512)
        for cand in candidates:
            # Check correctness via rule/verifier
            # Candidate contains code solution containing assertion checks at the bottom
            if run_code_verifier(cand):
                dataset.append({"prompt": prompt, "completion": cand})
                break # Keep the first correct path, or keep all to boost dataset
    
    with open("rft_sft_dataset.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    prompts = ["def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\nprint('AssertPassed')"]
    # Run pipeline mock logic:
    print("Pre-compiled RFT dataset ready for standard SFT training pipeline.")
```

## 5. Failure Modes & Recovery
- **Zero Correct Rollouts (Search Collapse):**
  - *Indicator:* For complex tasks, the model fails to generate even a single correct completion out of $K$ samples, leaving the dataset empty.
  - *Mitigation:* Increase sample size $K$, provide few-shot exemplars in the prompt, or initialize RFT from a model that has already undergone basic SFT.
- **Correctness-by-Chance / False Positives:**
  - *Indicator:* The verifier flags a completion as correct, but the model used faulty reasoning or shortcut patterns that don't generalize.
  - *Mitigation:* Implement rigorous, multi-test execution verifiers (test suite checking edge cases) rather than simple single assert checks.
