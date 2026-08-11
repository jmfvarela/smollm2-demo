"""
FastAPI server exposing an OpenAI-compatible chat completions endpoint
for SmolLM2 with streaming SSE support.

Exposes /v1/chat/completions — same format as the OpenAI API.
Compatible with NextChat, Open WebUI, and any OpenAI-compatible client.

Usage:
  python server.py
  # or via Docker:
  # docker compose up -d --build

Default model: HuggingFaceTB/SmolLM2-360M-Instruct (configurable via
SMOLLM2_MODEL_ID env var). Also works with SmolLM2-135M and other sizes.
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from model import Config, SmolLM2ForCausalLM

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = os.getenv("SMOLLM2_MODEL_ID", "HuggingFaceTB/SmolLM2-360M-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PROMPT_TOKENS = 256

# ── Global state ──────────────────────────────────────────────────────────────

model: SmolLM2ForCausalLM | None = None
tokenizer: AutoTokenizer | None = None


# ── Model loading (once at startup) ───────────────────────────────────────────

def load_model():
    global model, tokenizer

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading weights: {MODEL_ID} (bfloat16)")
    from transformers import AutoModelForCausalLM

    cfg = Config.from_hf(MODEL_ID)
    # Create model in bfloat16 to save RAM (vs float32)
    model = SmolLM2ForCausalLM(cfg).to(dtype=torch.bfloat16)

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
    )
    model.load_state_dict(hf_model.state_dict())
    model.lm_head.weight = model.model.embed_tokens.weight  # weight tying
    model.eval().to(DEVICE)

    del hf_model

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready: {n_params:,} parameters on {DEVICE}")


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="SmolLM2 API", version="0.1.0", lifespan=lifespan)


# ── OpenAI-compatible schemas ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "smollm2"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=128, ge=1, le=256)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class ChatMessageResponse(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ── Helper: build prompt from messages ────────────────────────────────────────

def build_prompt(messages: list[ChatMessage]) -> str:
    """
    Build a text prompt from a list of chat messages.
    Uses the tokenizer's chat_template if available (instruct models),
    falls back to a plain-text format for base models.
    """
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback for base models
    parts = []
    for m in messages:
        role = m.role.capitalize()
        parts.append(f"{role}: {m.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


# ── Non-streaming response ────────────────────────────────────────────────────

def _generate_response(req: ChatCompletionRequest, chat_id: str, created: int):
    msgs = list(req.messages)
    prompt = build_prompt(msgs)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_tokens = input_ids.shape[1]

    while prompt_tokens > MAX_PROMPT_TOKENS and len(msgs) > 1:
        msgs = msgs[2:]
        prompt = build_prompt(msgs)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        prompt_tokens = input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_tokens = output_ids.shape[1] - prompt_tokens
    new_tokens = output_ids[0, prompt_tokens:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    elapsed = time.time() - t0
    print(f"  ← {prompt_tokens}t + {completion_tokens}t in {elapsed:.1f}s "
          f"({completion_tokens/elapsed:.1f} tok/s)" if elapsed > 0 else "")

    return ChatCompletionResponse(
        id=chat_id, created=created, model=req.model,
        choices=[Choice(index=0, message=ChatMessageResponse(content=response_text.strip()),
                       finish_reason="stop")],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens),
    )


# ── Streaming SSE (Server-Sent Events) response ───────────────────────────────

def _generate_stream(req: ChatCompletionRequest, chat_id: str, created: int):
    msgs = list(req.messages)
    prompt = build_prompt(msgs)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_tokens = input_ids.shape[1]

    while prompt_tokens > MAX_PROMPT_TOKENS and len(msgs) > 1:
        msgs = msgs[2:]
        prompt = build_prompt(msgs)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        prompt_tokens = input_ids.shape[1]

    def sse_generator():
        # First chunk: assistant role
        first = {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": req.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n"

        t0 = time.time()
        completion_tokens = 0

        with torch.no_grad():
            for token_id in model.generate_stream(
                input_ids, max_new_tokens=req.max_tokens,
                temperature=req.temperature, top_p=req.top_p,
                eos_token_id=tokenizer.eos_token_id,
            ):
                completion_tokens += 1
                text = tokenizer.decode([token_id.item()], skip_special_tokens=True)

                chunk = {
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": text},
                                "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        elapsed = time.time() - t0
        print(f"  ← {prompt_tokens}t + {completion_tokens}t stream in {elapsed:.1f}s "
              f"({completion_tokens/elapsed:.1f} tok/s)" if elapsed > 0 else "")

        # Final chunk
        final = {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        return _generate_stream(req, chat_id, created)

    return _generate_response(req, chat_id, created)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


# ── Direct run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
