# SmolLM2-360M from Scratch

A didactic implementation of the LLaMA architecture with real weight loading
from HuggingFace. Built to study how a modern transformer works under the hood.

```bash
# Model only (study / demo)
python model.py

# Full API + Frontend (chat from mobile)
docker compose --profile smollm2 up -d --build
# Open https://chicaha.com/chat/
```

## How weights are downloaded and loaded

The model does **not include weights in the repo** (~270 MB). They are
downloaded automatically from HuggingFace Hub on first run:

1. `AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")`
   → downloads official weights, caches them in `~/.cache/huggingface/`
2. `AutoTokenizer.from_pretrained(...)` → downloads the tokenizer (vocabulary)
3. `custom_model.load_state_dict(hf_model.state_dict())` → copies weights
   to our implementation. **No manual mapping**: our module structure uses
   the exact same names as HuggingFace (`model.layers.0.self_attn.q_proj`,
   `model.embed_tokens`, etc.), so `load_state_dict()` works directly.

Subsequent runs use the local cache → no re-download.

## Files

| File | Purpose |
|------|---------|
| `model.py` | Full documented implementation (~430 lines). Importable, runnable. |
| `server.py` | FastAPI with `/v1/chat/completions` (OpenAI-compatible) + SSE streaming |
| `Dockerfile` | Python 3.13 container with CPU-only Torch (~675 MB) |
| `requirements.txt` | Python dependencies |

## Architecture

```
tokens → Embeddings → TransformerBlock × 30 → RMSNorm → lm_head → logits

Each TransformerBlock:
  RMSNorm → Attention (GQA + RoPE + KV-Cache) → (+) residual
  RMSNorm → SwiGLU MLP → (+) residual
```

## Components implemented

| Component | Detail |
|-----------|--------|
| Normalization | RMSNorm (no re-centering, faster than LayerNorm) |
| Positioning | RoPE (rotary, captures relative position with no extra params) |
| Attention | GQA 9:3 with KV-cache (3× compression factor) |
| MLP | SwiGLU (SiLU-gated, more expressive than ReLU) |
| KV-Cache | Stores K,V per layer. Prefill + incremental decode. ~3× faster |
| Streaming | SSE (`text/event-stream`) token by token, OpenAI-compatible |
| Loading | `load_state_dict()` direct (same names as HuggingFace) |

## Frontend (NextChat)

Starting with `--profile smollm2` also launches NextChat
at `https://chicaha.com/chat/`. It's a PWA: can be installed on mobile
as a native app.

Automatic config: model `smollm2-360m`, connected to the local API.

## Performance

360M parameters (~720 MB in bfloat16, ~1.4 GB in float32). With KV-cache and SSE streaming:

- **3-5 tok/s** on CPU (Hetzner 2 vCPU) in bfloat16
- Typical latency: 4-8 seconds for short responses
- Auto-detects config (layers, heads, dims) from HuggingFace — no hardcoded values
- Without KV-cache it would be 2-3× slower
