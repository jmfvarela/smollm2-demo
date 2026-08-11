# SmolLM2 Demo

A didactic implementation of the LLaMA architecture with real weight loading
from HuggingFace. Built to study how a modern transformer works under the hood.

## Quick Start

```bash
git clone git@github.com:jmfvarela/smollm2-demo.git
cd smollm2-demo

# One command → API + Chat UI
docker compose up -d --build
```

Then open **http://localhost:3001** and start chatting.

> ⚠️ First run downloads ~270 MB of model weights from HuggingFace
> and takes 2-5 minutes. Subsequent starts are instant (cached).

The API is at http://localhost:3000 if you want to call it directly:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm2-360m","messages":[{"role":"user","content":"Hello"}]}'
```

## Study the model (no Docker)

```bash
pip install -r requirements.txt
python model.py
```

## How weights are downloaded

The model does **not include weights in the repo** (~270 MB). They are
downloaded automatically from HuggingFace Hub on first run:

1. `AutoModelForCausalLM.from_pretrained(...)` → downloads weights, caches in `~/.cache/huggingface/`
2. `AutoTokenizer.from_pretrained(...)` → downloads tokenizer
3. `custom_model.load_state_dict(hf_model.state_dict())` → copies weights to our implementation. **No manual mapping** — same module names as HuggingFace.

Subsequent runs use local cache → no re-download.

## Files

| File | Purpose |
|------|---------|
| `model.py` | Full documented implementation (465 lines). Importable, runnable. |
| `server.py` | FastAPI with `/v1/chat/completions` (OpenAI-compatible) + SSE streaming |
| `Dockerfile` | Python 3.13 container with CPU-only Torch |
| `requirements.txt` | Python dependencies |
| `docker-compose.yml` | smollm2-seed + NextChat + optional nginx |
| `nginx.conf` | Reverse proxy config for the optional nginx service |

## Architecture

```
tokens → Embeddings → TransformerBlock × 30 → RMSNorm → lm_head → logits

Each TransformerBlock:
  RMSNorm → Attention (GQA + RoPE + KV-Cache) → (+) residual
  RMSNorm → SwiGLU MLP → (+) residual
```

## Components

| Component | Detail |
|-----------|--------|
| Normalization | RMSNorm (no re-centering, faster than LayerNorm) |
| Positioning | RoPE (rotary, captures relative position) |
| Attention | GQA 9:3 with KV-cache (3× compression) |
| MLP | SwiGLU (SiLU-gated, more expressive than ReLU) |
| KV-Cache | Stores K,V per layer. Prefill + incremental decode. ~3× faster |
| Streaming | SSE (`text/event-stream`) token by token, OpenAI-compatible |
| Loading | `load_state_dict()` direct (same names as HuggingFace) |

## Frontend (NextChat)

`docker compose up -d` launches both the API server and NextChat Web UI.
NextChat is a PWA — can be installed on mobile as a native app.

Automatic config: model `smollm2-360m`, connected to the local API.

## Performance

360M parameters (~720 MB in bfloat16, ~1.4 GB in float32). With KV-cache and SSE streaming:

- **3-5 tok/s** on CPU (2 vCPU) in bfloat16
- Typical latency: 4-8 seconds for short responses
- Auto-detects config (layers, heads, dims) from HuggingFace
- Without KV-cache it would be 2-3× slower
