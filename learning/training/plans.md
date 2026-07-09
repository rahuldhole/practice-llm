# Training plans

### I. The "From Scratch" Paradigms (Foundational)

| Paradigm | Description | When to Train & Why (Use Cases) |
| :--- | :--- | :--- |
| **1. [Dense Pre-training](I_foundational/dense_pretraining.md)** | Classic Chinchilla-optimal scaling. | Use when building a custom foundation model from scratch with a large budget ($50k+) and proprietary data. Ensures maximum scaling stability. |
| **2. [Sparse (MoE) Pre-training](I_foundational/sparse_moe_pretraining.md)** | Training the router and experts from random initialization. | Use when you need the intelligence of a large model (8B+) but have a strict inference speed/compute budget (active params <2B). |
| **3. [Long-Context Pre-training](I_foundational/long_context.md)** | Focus on architectural modifications (RoPE scaling, Ring Attention) to support 128k+ tokens. | Use when adapting a model to handle huge documents, entire codebases, or complex multi-turn chats natively. |
| **4. [Tokenizer-First Training](I_foundational/tokenizer_bespoke.md)** | Building a bespoke BPE/SentencePiece vocabulary to optimize for a specific language or domain (e.g., Code/Math). | Use when working with highly specialized syntax (logs, chemical formulas, code). Standard tokenizers fragment these, wasting context window. |

### II. Adaptation Paradigms (The "Moat" Builders)

| Paradigm | Description | When to Train & Why (Use Cases) |
| :--- | :--- | :--- |
| **5. [Continual Pre-training (CPT)](II_adaptation/continual_pretraining.md)** | Domain-injection into a generic base (e.g., Llama/Qwen). | Use when you need to inject deep domain knowledge (legal, medical, corporate docs) into an already smart base model. Saves 95%+ compute vs. scratch. |
| **6. [Model Merging (Frankenmerging)](II_adaptation/frankenmerging.md)** | Combining two or more fine-tuned models (e.g., Slerp, DARE, TIES) without retraining. | Use when you want to merge separate capabilities (e.g., coding + creative writing) instantly with zero training cost. |
| **7. [Downcycling](II_adaptation/downcycling_moe.md)** | Converting a Dense model into an MoE by cloning FFN layers. | Use when upgrading a highly-capable Dense model to a faster MoE architecture, avoiding expensive random-init pre-training. |
| **8. [Knowledge Distillation](II_adaptation/knowledge_distillation.md)** | Training a "Student" (3B) to mimic a "Teacher" (405B/GPT-4o) using KL-Divergence. | Use when targeting edge/mobile deployments. Transfers teacher log probabilities to achieve outsized reasoning power in a lightweight model. |

### III. Alignment & Preference Paradigms (Post-Training)

| Paradigm | Description | When to Train & Why (Use Cases) |
| :--- | :--- | :--- |
| **9. [SFT (Supervised Fine-Tuning)](III_alignment/supervised_finetuning.md)** | Standard Instruction-Following data. | First post-training step to transition a base pre-trained model into an instruction-following assistant. Teaches formatting and prompt response. |
| **10. [DPO (Direct Preference Optimization)](III_alignment/direct_preference_opt.md)** | The current standard for "voice" and "refusal" alignment. | Use when aligning model tone, personality, or safety guardrails. Eliminates the need to train a separate reward model (much more stable than PPO). |
| **11. [ORPO (Odds Ratio Preference Optimization)](III_alignment/odds_ratio_preference.md)** | Aligning in a single stage (SFT + DPO merged). | Use to speed up alignment and prevent the model from losing critical token probabilities during disjoint SFT and preference phases. |
| **12. [KTO (Kahneman-Tversky Optimization)](III_alignment/kahneman_tversky_opt.md)** | Alignment using binary (Good/Bad) feedback instead of pairs. | Use when collecting thumbs-up/down feedback from production users. Easier to harvest than structured pairwise preferences. |
| **13. [RLHF (PPO/GRPO)](III_alignment/reinforcement_learning_ppo.md)** | Traditional Reinforcement Learning using a Reward Model. | Use for complex logical tasks (math, coding, reasoning loops) where outcomes can be validated by rules or programmatic verifiers. |

### IV. Efficiency Paradigms (Resource-Constrained)

| Paradigm | Description | When to Train & Why (Use Cases) |
| :--- | :--- | :--- |
| **14. [PEFT (LoRA/QLoRA)](IV_efficiency/peft_lora_qlora.md)** | Low-Rank Adaptation; updating <1% of params. | Use when training on restricted hardware (single GPU) or when deploying thousands of task-specific adapters for multi-tenant apps. |
| **15. [Q-Fine-Tuning](IV_efficiency/q_finetuning.md)** | Fine-tuning in 4-bit or 8-bit precision. | Use to fit training of large models (e.g., 70B) onto accessible consumer/indie hardware by compressing parameter representation. |
| **16. [Adapter Training](IV_efficiency/adapter_bottlenecks.md)** | Adding external "bottleneck" layers rather than modifying base weights. | Use when modular plug-and-play capability is required at runtime without modifying or altering the underlying base model parameters. |
| **17. [Quantization-Aware Training (QAT)](IV_efficiency/quantization_aware_training.md)** | Training the model specifically to withstand heavy compression (e.g., GGUF/EXL2). | Use when targeting ultra-low bitrates (2-bit or 3-bit) for edge deployment, preventing the heavy accuracy drops of post-training quantization. |

### V. Data-Centric Paradigms (The "Creative" Moats)

| Paradigm | Description | When to Train & Why (Use Cases) |
| :--- | :--- | :--- |
| **18. [Synthetic Data Scaling](V_datacentric/synthetic_data_scaling.md)** | Using a "Teacher" to generate 100x your original dataset size. | Use when human-labeled data is scarce, expensive, or hard to scale. Allows you to train a smaller model to near-frontier reasoning performance. |
| **19. [Curriculum Learning](V_datacentric/curriculum_learning.md)** | Training on "Easy to Hard" data sequences. | Use for highly logical and structured tasks (math, coding). Ramping up difficulty systematically prevents gradient confusion and speeds convergence. |
| **20. [Reflection/Self-Correction Training](V_datacentric/self_correction_reflection.md)** | Training the model to generate a "thought" step, evaluate its own logic, and rewrite. | Use to build reliable reasoning models. Teaches the model to debug its own logic prior to spitting out the final answer. |
| **21. [Multi-Modal Injection](V_datacentric/multimodal_projection.md)** | Projecting Vision/Audio encoders into the latent space (e.g., CLIP-projection). | Use when adding image, audio, or video capabilities to an existing text model, avoiding training the entire system from scratch. |

### VI. Architectural Modification Paradigms

| Paradigm | Description | When to Train & Why (Use Cases) |
| :--- | :--- | :--- |
| **22. [Weight Pruning](VI_architectural/weight_pruning.md)** | Removing redundant heads or layers post-training. | Use when optimizing a model for faster inference speeds and smaller disk footprints by discarding less-activated channels or weights. |
| **23. [Depth-to-Width Conversion](VI_architectural/depth_width_conversion.md)** | Changing the layer count and hidden dimension ratios. | Use to tune a model architecture for specific hardware compute-to-bandwidth ratios (e.g., sequential scaling bottlenecks). |
| **24. [Activations Tuning](VI_architectural/activation_tuning.md)** | Adjusting SwiGLU/GeLU activation functions during fine-tuning. | Use when fine-tuning a model for optimal latency throughput or when seeking slight training stability gains from activation behavior. |

### VII. Operational/Infrastructure Paradigms

| Paradigm | Description | When to Train & Why (Use Cases) |
| :--- | :--- | :--- |
| **25. [Federated Training](VII_operational/federated_training.md)** | Training across multiple independent, non-sharing data silos. | Use when training on highly restricted or private data (medical logs, defense files) spread across remote clients that cannot share raw datasets. |
| **26. [Active Learning](VII_operational/active_learning_loops.md)** | Iterative loops where the model "chooses" which data it finds most confusing for manual labeling. | Use when human annotation is extremely expensive. Saves labeling costs by only requesting labels for items the model is highly uncertain about. |

---

### How to use this:

* **$100 Plan:** ([Plan 5](II_adaptation/continual_pretraining.md) + [Plan 14](IV_efficiency/peft_lora_qlora.md) + [Plan 9](III_alignment/supervised_finetuning.md)).
* **Indie MoE Plan:** ([Plan 7](II_adaptation/downcycling_moe.md) + [Plan 2](I_foundational/sparse_moe_pretraining.md) + [Plan 10](III_alignment/direct_preference_opt.md)).
* **Master Plan:** ([Plan 1](I_foundational/dense_pretraining.md) + [Plan 8](II_adaptation/knowledge_distillation.md) + [Plan 18](V_datacentric/synthetic_data_scaling.md) + [Plan 20](V_datacentric/self_correction_reflection.md) + [Plan 11](III_alignment/odds_ratio_preference.md)).


