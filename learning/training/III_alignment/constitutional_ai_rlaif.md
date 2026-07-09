# Blueprint: Constitutional AI (RLAIF)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Aligns model behavior without human feedback using a small set of principles (a "constitution").
  - **Phase 1 (Supervised Stage):** Critique and revise initial model outputs using prompts containing constitutional guidelines.
  - **Phase 2 (Reinforcement/Preference Learning):** Train a preference classifier (or perform DPO directly) on pairs of revised vs. unrevised model outputs to align the student model.
- **FLOPs Scaling Formula:**
  - Cost is dominated by prompt-token generations (API queries or local teacher model rollouts).
  - Aligns student model via direct preference optimization on generated dataset:
    $$\text{FLOPs} \approx 9 \times N_{\text{student}} \times D_{\text{pairs}}$$
- **Precision Profile:**
  - **Teacher Model (Critique/Revision):** FP16/BF16 (or high-precision API).
  - **Student Model (DPO/ORPO):** BF16.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM) or API access (OpenAI/Anthropic).
- **Provider:** RunPod or API.
- **Execution Duration Limit:** < 8 hours.
- **Target Token/Batch Scale:** 10,000 critique/revision pairs.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 SXM5 nodes.
- **Interconnect Requirements:** NVLink (intra-node) for model evaluations.
- **Persistent Storage Topography:** Centralized PostgreSQL database to cache prompts, critiques, revisions, and alignment preferences.

## 3. Data Topography
- **Token Window Length:** 4,092 tokens.
- **Preprocessing Requirements:** Generate a diverse list of safety and capability test prompts. Critique step adds instructions like: *"Identify if the response is harmful or biased based on Rule X of the constitution..."*
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Script simulating the critique and revision steps of Constitutional AI.

import openai

CONSTITUTION = [
    "Choose the response that is most helpful, honest, and harmless.",
    "Avoid toxic, insulting, condescending, or biased language."
]

def generate_constitutional_revision(prompt, base_response):
    # Step 1: Critique
    critique_prompt = f"""
    Here is a prompt: {prompt}
    Here is a response: {base_response}
    
    Critique this response using the principle: "{CONSTITUTION[1]}"
    Discuss any potential harm, bias, or toxic elements.
    """
    
    # Run critique via LLM API
    critique = "Critique: The response uses slightly biased assumptions..." # Mock output
    
    # Step 2: Revision
    revision_prompt = f"""
    Prompt: {prompt}
    Response: {base_response}
    Critique: {critique}
    
    Rewrite the response to address the critique, ensuring it follows the principle: "{CONSTITUTION[0]}".
    """
    
    # Run revision via LLM API
    revised_response = "Revised Response: [Harm-free text]" # Mock output
    
    return base_response, revised_response

if __name__ == "__main__":
    prompt = "Tell me why some groups are worse at math."
    bad_output = "Some groups lack analytical features in their historical education..."
    orig, rev = generate_constitutional_revision(prompt, bad_output)
    print("Original:", orig)
    print("Revised:", rev)
```

## 5. Failure Modes & Recovery
- **"Helpfulness vs. Harmlessness" Tradeoff (Sycophancy/Refusals):**
  - *Indicator:* The model over-generalizes safety guidelines, refusing to answer benign prompts (e.g. *"I cannot tell you how to kill a command line process..."*).
  - *Mitigation:* Balance the constitution with explicit instructions to prioritize helpfulness for non-malicious prompts. Add positive examples in SFT.
- **Critic Model Collapse:**
  - *Indicator:* The critic model starts repeating boilerplate critiques or outputting trivial edits that don't improve response quality.
  - *Mitigation:* Use a larger, more capable model (e.g. Claude 3 Opus / GPT-4o) as the critic/revisor, or utilize few-shot exemplars in the critique prompt.
