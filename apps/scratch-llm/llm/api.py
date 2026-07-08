from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
from llm.model import TinyLLM
from llm.generate import load_tokenizer, generate_text

app = FastAPI(title="Scratch LLM API")

# Setup global variables
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 10

@app.on_event("startup")
def startup_event():
    global model, tokenizer
    d_model = 16
    
    if not os.path.exists("dist/vocab.json") or not os.path.exists("dist/model.pth"):
        print("Model or vocab not found. Please run train.py first.")
        return

    # Load tokenizer
    tokenizer = load_tokenizer("dist/vocab.json")
    vocab_size = tokenizer.vocab_size
    
    # Initialize model and load weights
    model = TinyLLM(vocab_size, d_model)
    model.load_state_dict(torch.load("dist/model.pth"))
    model.eval()
    print("Model loaded successfully.")

@app.post("/generate")
def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Train the model first.")
        
    try:
        result = generate_text(model, tokenizer, req.prompt, max_new_tokens=req.max_new_tokens)
        return {"prompt": req.prompt, "generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
