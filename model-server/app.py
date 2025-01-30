from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from model import SmolLM2ForCausalLM
from config import SmolLM2Config
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmolLM2 Shakespeare API",
    description="API for generating Shakespeare-style text",
    version="1.0.0"
)

# Store last generated result
last_result = {"prompt": "", "generated_text": ""}

# Add middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7

def load_model():
    try:
        config = SmolLM2Config()
        model = SmolLM2ForCausalLM(config)
        
        checkpoint = torch.load('checkpoints/step_5050.pt', map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.model.'):
                new_key = key.replace('model.model.', 'model.')
                new_state_dict[new_key] = value
            elif key.startswith('model.lm_head.'):
                new_key = key.replace('model.lm_head.', 'lm_head.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/cosmo2-tokenizer",
            revision=None,
            use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model, tokenizer = load_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <html>
        <head>
            <title>SmolLM2 Shakespeare API</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .result {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>SmolLM2 Shakespeare API</h1>
            <div class="result">
                <h2>Last Generated Text:</h2>
                <p><b>Prompt:</b> {last_result["prompt"]}</p>
                <p><b>Generated:</b> {last_result["generated_text"]}</p>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

@app.post("/generate")
async def generate(request: GenerationRequest):
    global last_result
    logger.info(f"Generate endpoint accessed with prompt: {request.prompt[:50]}...")
    try:
        with torch.no_grad():
            input_ids = tokenizer(
                request.prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )["input_ids"]
            
            output_sequence = input_ids.clone()
            
            for _ in range(request.max_tokens):
                outputs, _ = model(input_ids=output_sequence)
                next_token_logits = outputs[:, -1, :] / request.temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                output_sequence = torch.cat([output_sequence, next_token], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
            generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
            last_result = {
                "prompt": request.prompt,
                "generated_text": generated_text
            }
            return {"generated_text": generated_text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 