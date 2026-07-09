# Training plans

### I. The "From Scratch" Paradigms (Foundational)

* **1. [Dense Pre-training](I_foundational/dense_pretraining.md):** Classic Chinchilla-optimal scaling.
* **2. [Sparse (MoE) Pre-training](I_foundational/sparse_moe_pretraining.md):** Training the router and experts from random initialization.
* **3. [Long-Context Pre-training](I_foundational/long_context.md):** Focus on architectural modifications (RoPE scaling, Ring Attention) to support 128k+ tokens.
* **4. [Tokenizer-First Training](I_foundational/tokenizer_bespoke.md):** Building a bespoke BPE/SentencePiece vocabulary to optimize for a specific language or domain (e.g., Code/Math).

### II. Adaptation Paradigms (The "Moat" Builders)

* **5. [Continual Pre-training (CPT)](II_adaptation/continual_pretraining.md):** Domain-injection into a generic base (e.g., Llama/Qwen).
* **6. [Model Merging (Frankenmerging)](II_adaptation/frankenmerging.md):** Combining two or more fine-tuned models (e.g., Slerp, DARE, TIES) to merge capabilities without retraining.
* **7. [Downcycling](II_adaptation/downcycling_moe.md):** Converting a Dense model into an MoE by cloning FFN layers.
* **8. [Knowledge Distillation](II_adaptation/knowledge_distillation.md):** Training a "Student" (3B) to mimic a "Teacher" (405B/GPT-4o) using KL-Divergence.

### III. Alignment & Preference Paradigms (Post-Training)

* **9. [SFT (Supervised Fine-Tuning)](III_alignment/supervised_finetuning.md):** Standard Instruction-Following data.
* **10. [DPO (Direct Preference Optimization)](III_alignment/direct_preference_opt.md):** The current standard for "voice" and "refusal" alignment.
* **11. [ORPO (Odds Ratio Preference Optimization)](III_alignment/odds_ratio_preference.md):** Aligning in a single stage (SFT + DPO merged).
* **12. [KTO (Kahneman-Tversky Optimization)](III_alignment/kahneman_tversky_opt.md):** Alignment using binary (Good/Bad) feedback instead of pairs.
* **13. [RLHF (PPO/GRPO)](III_alignment/reinforcement_learning_ppo.md):** Traditional Reinforcement Learning using a Reward Model.

### IV. Efficiency Paradigms (Resource-Constrained)

* **14. [PEFT (LoRA/QLoRA)](IV_efficiency/peft_lora_qlora.md):** Low-Rank Adaptation; updating <1% of params.
* **15. [Q-Fine-Tuning](IV_efficiency/q_finetuning.md):** Fine-tuning in 4-bit or 8-bit precision.
* **16. [Adapter Training](IV_efficiency/adapter_bottlenecks.md):** Adding external "bottleneck" layers rather than modifying base weights.
* **17. [Quantization-Aware Training (QAT)](IV_efficiency/quantization_aware_training.md):** Training the model specifically to withstand heavy compression (e.g., GGUF/EXL2).

### V. Data-Centric Paradigms (The "Creative" Moats)

* **18. [Synthetic Data Scaling](V_datacentric/synthetic_data_scaling.md):** Using a "Teacher" to generate 100x your original dataset size.
* **19. [Curriculum Learning](V_datacentric/curriculum_learning.md):** Training on "Easy to Hard" data sequences.
* **20. [Reflection/Self-Correction Training](V_datacentric/self_correction_reflection.md):** Training the model to generate a "thought" step, evaluate its own logic, and rewrite.
* **21. [Multi-Modal Injection](V_datacentric/multimodal_projection.md):** Projecting Vision/Audio encoders into the latent space (e.g., CLIP-projection).

### VI. Architectural Modification Paradigms

* **22. [Weight Pruning](VI_architectural/weight_pruning.md):** Removing redundant heads or layers post-training.
* **23. [Depth-to-Width Conversion](VI_architectural/depth_width_conversion.md):** Changing the layer count and hidden dimension ratios.
* **24. [Activations Tuning](VI_architectural/activation_tuning.md):** Adjusting SwiGLU/GeLU activation functions during fine-tuning.

### VII. Operational/Infrastructure Paradigms

* **25. [Federated Training](VII_operational/federated_training.md):** Training across multiple independent, non-sharing data silos.
* **26. [Active Learning](VII_operational/active_learning_loops.md):** Iterative loops where the model "chooses" which data it finds most confusing, which you then label manually.

---

### How to use this:

* **$100 Plan:** ([Plan 5](II_adaptation/continual_pretraining.md) + [Plan 14](IV_efficiency/peft_lora_qlora.md) + [Plan 9](III_alignment/supervised_finetuning.md)).
* **Indie MoE Plan:** ([Plan 7](II_adaptation/downcycling_moe.md) + [Plan 2](I_foundational/sparse_moe_pretraining.md) + [Plan 10](III_alignment/direct_preference_opt.md)).
* **Master Plan:** ([Plan 1](I_foundational/dense_pretraining.md) + [Plan 8](II_adaptation/knowledge_distillation.md) + [Plan 18](V_datacentric/synthetic_data_scaling.md) + [Plan 20](V_datacentric/self_correction_reflection.md) + [Plan 11](III_alignment/odds_ratio_preference.md)).

