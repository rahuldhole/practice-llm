import argparse
import os
import torch
import gradio as gr
from llm.model import ScratchLLM
from llm.generate import load_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Run ScratchLLM Gradio Chat UI")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu, cuda, mps)")
    args = parser.parse_args()

    # Parameters (Must match training config)
    d_model = 64
    n_heads = 4
    n_kv_heads = 2
    n_layers = 4

    # Device configuration
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
        
    print(f"Running inference on device: {device}")

    # Paths
    model_path = "dist/model.pth"
    vocab_path = "dist/tokenizer.json"

    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(
            "Model checkpoint or tokenizer not found. "
            "Please run training first to generate them (e.g., `task train`)."
        )

    # Load tokenizer and model
    tokenizer = load_tokenizer(vocab_path)
    vocab_size = tokenizer.vocab_size

    model = ScratchLLM(vocab_size, d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, n_layers=n_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    def predict(message, history, system_prompt, temperature, top_k, max_new_tokens):
        # Format the chat history into a model prompt
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n"
        for user_msg, bot_msg in history:
            prompt += f"User: {user_msg}\nAI: {bot_msg}\n"
        prompt += f"User: {message}\nAI:"

        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        
        from llm.model import KVCache
        kv_cache = KVCache()
        
        generated_ids = []
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                if i == 0:
                    logits = model(x, start_pos=0, kv_cache=kv_cache)
                else:
                    logits = model(x[:, -1:], start_pos=x.shape[1] - 1, kv_cache=kv_cache)
                
                next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)
                
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[-1]] = -float('Inf')
                    
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                if next_token_id == tokenizer.pad_token_id:
                    break
                    
                generated_ids.append(next_token_id)
                x = torch.cat([x, torch.tensor([[next_token_id]], device=device)], dim=1)
                
                # Yield current string
                yield tokenizer.decode(generated_ids)

    # Custom CSS for design aesthetics
    custom_css = """
    body {
        background: radial-gradient(circle at top, #1a1e29, #0d0f14) !important;
        font-family: 'Outfit', 'Inter', sans-serif !important;
    }
    .gradio-container {
        border: none !important;
        max-width: 1100px !important;
        margin: 40px auto !important;
    }
    .header-banner {
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px -10px rgba(79, 70, 229, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-banner h1 {
        color: white !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        letter-spacing: -0.05em;
    }
    .header-banner p {
        color: #e2e8f0 !important;
        font-size: 1.2rem !important;
        font-weight: 400;
    }
    .main-grid {
        gap: 1.5rem !important;
    }
    .chat-box {
        background: rgba(22, 28, 45, 0.6) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 0.5rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
    }
    .settings-panel {
        background: rgba(22, 28, 45, 0.4) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1) !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #4f46e5, #06b6d4) !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.4) !important;
    }
    """

    # Build Gradio UI
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate"
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="*neutral_900",
        block_border_width="1px",
        block_border_color="*neutral_800",
        block_radius="12px"
    )

    with gr.Blocks(theme=theme, css=custom_css, title="ScratchLLM Chat UI") as demo:
        with gr.Div(elem_classes="header-banner"):
            gr.HTML("<h1>ScratchLLM Interactive Chat</h1>")
            gr.HTML("<p>Interact with a tiny GPT-style model featuring SOTA Pre-Norm RMSNorm, SwiGLU activations, GQA, FlashAttention, and KV Caching.</p>")

        with gr.Row(elem_classes="main-grid"):
            with gr.Column(scale=3, elem_classes="chat-box"):
                chatbot = gr.ChatInterface(
                    fn=predict,
                    additional_inputs=[
                        gr.Textbox(
                            value="You are ScratchLLM, a helpful and polite AI assistant.",
                            label="System Prompt",
                            placeholder="Set assistant persona...",
                            lines=2
                        ),
                        gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="Higher values produce more creative output, lower values are more deterministic."
                        ),
                        gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=10,
                            step=1,
                            label="Top-K",
                            info="Filter logits to only the top K tokens. Set to 0 to disable."
                        ),
                        gr.Slider(
                            minimum=5,
                            maximum=128,
                            value=50,
                            step=5,
                            label="Max Gen Tokens",
                            info="Maximum number of new tokens to generate."
                        )
                    ]
                )
                
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
