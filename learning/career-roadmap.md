# 💼 Career Roadmap: Landing a Job as an LLM/AI Engineer

Use this roadmap as a checklist for your professional portfolio. The goal is to build, evaluate, and optimize production-ready LLM pipelines.

---

## 🛠️ Essential Industry Stack to Master
Make sure you gain hands-on experience with these standard tools:
- [ ] **Training / Finetuning**: `transformers`, `trl` (SFTTrainer, DPOTrainer), `axolotl`, `DeepSpeed` (ZeRO 2/3)
- [ ] **PEFT / Quantization**: `peft` (LoRA/QLoRA), `bitsandbytes`, `AutoGPTQ`
- [ ] **Inference / Serving**: `vLLM`, `Ollama`, `llama.cpp`
- [ ] **Evaluation**: `lm-evaluation-harness`, `deepspeed-eval`

---

## 📋 Progression Checklist & Portfolio Projects

### Phase 1: Instruction Tuning & Efficiency
- [ ] **Supervised Fine-Tuning (SFT)**
  - [ ] Implement prompt formatting templates (ChatML, Llama-3 style).
  - [ ] Apply response-only loss masking (masking out prompt tokens in target labels).
- [ ] **PEFT (LoRA & QLoRA)**
  - [ ] QLoRA fine-tune a 1B–8B model (e.g., `Qwen2.5-1.5B` or `Llama-3-8B`) in 4-bit precision.
  - [ ] **Portfolio Project 1**: *Domain-Specific SFT Assistant* (e.g., fine-tuning a small model to write SQL query schemas or convert logs).

### Phase 2: Preference Alignment & Safety
- [ ] **Direct Preference Optimization (DPO/ORPO)**
  - [ ] Run a preference alignment script using a chosen/rejected dataset.
  - [ ] Compare SFT vs. DPO performance using human-in-the-loop or LLM-as-a-judge.
- [ ] **RLHF & Reasoning (GRPO)**
  - [ ] Understand DeepSeek-style GRPO (group relative rewards based on programmatic verifiers like code compilers or math equations).
  - [ ] **Portfolio Project 2**: *Self-Correcting Math Solver* (using GRPO/rejection-sampling to reward correct outputs).

### Phase 3: Synthetic Data & Continual Pretraining (CPT)
- [ ] **Synthetic Data Pipeline**
  - [ ] Write a script using a frontier API (or local host) to generate diverse training examples.
  - [ ] Implement deduplication and quality-filtering (perplexity filtering, embedding clustering).
- [ ] **Continual Pretraining**
  - [ ] Adapt a base model to a new language or heavy technical domain using raw text.
  - [ ] **Portfolio Project 3**: *Corporate Knowledge Injector* (CPT on a PDF/documentation corpus followed by SFT).

### Phase 4: Agentic Capabilities & Tool Integration
- [ ] **Tool Use / Function Calling**
  - [ ] Fine-tune a model to generate exact JSON payloads to interact with Python APIs.
  - [ ] **Portfolio Project 4**: *Local Agentic Tool Executor* (an agent that uses your fine-tuned model to run local terminal commands or query databases).

### Phase 5: Optimization & Production Evaluations
- [ ] **Knowledge Distillation**
  - [ ] Distill teacher logits into a smaller student architecture.
- [ ] **Evaluation Harnesses**
  - [ ] Set up `lm-evaluation-harness` to run local benchmark datasets.
  - [ ] **Portfolio Project 5**: *Custom Benchmarked LLM* (complete evaluation report comparing your custom models on target tasks).