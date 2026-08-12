"""
SmolLM2 from Scratch — LLaMA Architecture with KV-Cache
=============================================================

A minimal, didactic implementation of the LLaMA architecture used by
SmolLM2, LLaMA 2/3, DeepSeek-V3 (base), GLM-4, and Mistral.

Loads REAL weights from HuggingFace and runs end-to-end inference.

Components:
  1. RMSNorm              — pre-layer normalization without re-centering
  2. RoPE                 — rotary positional embeddings
  3. Grouped-Query Attn   — attention with KV-cache and fewer K-V heads than Q
  4. SwiGLU MLP           — feed-forward with SiLU-gated activation
  5. TransformerBlock     — pre-norm + residual connections + KV-cache passthrough
  6. Transformer          — embeddings + N blocks + final norm
  7. SmolLM2ForCausalLM   — full model with lm_head + generate with KV-cache

Uses the same submodule/parameter names as HuggingFace → load_state_dict()
works directly with no manual mapping.

References:
  LLaMA:   https://arxiv.org/abs/2302.13971
  RoPE:    https://arxiv.org/abs/2104.09864
  SwiGLU:  https://arxiv.org/abs/2002.05202
  GQA:     https://arxiv.org/abs/2305.13245
  KV-cache: https://arxiv.org/abs/2210.03057
  SmolLM2: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct (360M)
           https://huggingface.co/HuggingFaceTB/SmolLM2-135M (135M)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG — Default hyperparameters (auto-detected via Config.from_hf)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    # Defaults match SmolLM2-360M-Instruct. They are illustrative only —
    # Config.from_hf() overrides ALL fields from the actual HuggingFace model.
    vocab_size: int = 49152
    hidden_size: int = 960
    num_hidden_layers: int = 32
    num_attention_heads: int = 15
    num_key_value_heads: int = 5        # GQA ratio 3:1
    intermediate_size: int = 2560        # ≈ 8/3 × hidden_size
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    max_position_embeddings: int = 8192

    @classmethod
    def from_hf(cls, model_id: str) -> "Config":
        """Build Config from a HuggingFace model ID (auto-detects hyperparameters)."""
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(model_id)
        # rope_theta moved to rope_parameters dict in transformers 5.x
        theta = getattr(hf, "rope_theta", None)
        if theta is None and hasattr(hf, "rope_parameters"):
            theta = hf.rope_parameters.get("rope_theta", 100000.0)
        if theta is None:
            theta = 100000.0
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            intermediate_size=hf.intermediate_size,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=theta,
            max_position_embeddings=hf.max_position_embeddings,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RMSNorm
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ  — faster than LayerNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RoPE — Rotary Position Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
#
# Encodes absolute position as a 2D rotation in attention space.
# θ_i = 1 / (10000^(2i/d))  →  angle at position p = p × θ_i
#
# Key properties:
#   (1) Q_i·K_j depends only on (i−j), the RELATIVE position.
#   (2) Applied at the start of attention, no extra parameters.
#
# High frequencies (large θ) → fast variation → capture local position.
# Low frequencies (small θ) → slow variation → capture global position.


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 100000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs_emb = torch.outer(t, freqs)                     # (seq_len, dim/2)
    freqs_emb = torch.cat((freqs_emb, freqs_emb), dim=-1) # (seq_len, dim)
    return freqs_emb.cos(), freqs_emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GROUPED-QUERY ATTENTION with KV-CACHE
# ═══════════════════════════════════════════════════════════════════════════════
#
# KV-Cache is THE optimization that makes autoregressive inference viable.
#
# Without cache:  each new token → recompute attention over ENTIRE sequence
#                 cost at step t: O(t²)  →  total: O(n³) for n tokens
# With cache:     store K,V from previous steps → only compute for new token
#                 cost at step t: O(t)   →  total: O(n²) for n tokens
#
# In practice: 3-50× faster generation depending on sequence length.


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_q_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_q_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_pos: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          (B, S, D) — input. S=1 in decode mode, S>1 in prefill.
            cos, sin:   (total_len, head_dim) — precomputed RoPE for all positions.
            cache_pos:  starting position of these tokens in the full sequence.
            k_cache:    (B, n_kv, cache_pos, hd) — accumulated K from previous steps.
            v_cache:    (B, n_kv, cache_pos, hd) — accumulated V from previous steps.
        Returns:
            out:        (B, S, D)
            k_new:      (B, n_kv, cache_pos+S, hd) — updated K (to store in cache).
            v_new:      (B, n_kv, cache_pos+S, hd) — updated V.
        """
        B, S, _ = x.shape

        # Projections → (B, n_heads, S, head_dim)
        q = self.q_proj(x).view(B, S, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE: slice cos/sin for positions cache_pos .. cache_pos+S
        q, k = apply_rope(q, k, cos[cache_pos:cache_pos + S], sin[cache_pos:cache_pos + S])

        # KV-Cache: concatenate with accumulated K,V
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        k_new, v_new = k, v  # save to return to caller

        # GQA: expand K,V to match n_q_heads count
        if self.n_rep > 1:
            k_exp = k.repeat_interleave(self.n_rep, dim=1)
            v_exp = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_exp, v_exp = k, v

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = (q @ k_exp.transpose(-2, -1)) * scale  # (B, n_q, S, total_S)

        # Causal mask: only needed during prefill (S > 1, no prior cache).
        # During decode (S=1) no mask needed: the new token can attend to all cached tokens.
        if k_cache is None:
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v_exp  # (B, n_q, S, hd)

        # Recombine heads → (B, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), k_new, v_new


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SwiGLU MLP
# ═══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """SwiGLU(x) = W_down · (SiLU(W_gate·x) ⊙ W_up·x)"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRANSFORMER BLOCK (Pre-Norm)
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_pos: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, k_new, v_new = self.self_attn(
            self.input_layernorm(x), cos, sin, cache_pos, k_cache, v_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k_new, v_new


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TRANSFORMER — Embeddings + Layers + Final Norm
# ═══════════════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_pos: int = 0,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.embed_tokens(x)
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            k_cache, v_cache = kv_caches[i] if kv_caches else (None, None)
            x, k_new, v_new = layer(x, cos, sin, cache_pos, k_cache, v_cache)
            new_caches.append((k_new.detach(), v_new.detach()))

        return self.norm(x), new_caches


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FULL MODEL — SmolLM2ForCausalLM with KV-Cache
# ═══════════════════════════════════════════════════════════════════════════════

class SmolLM2ForCausalLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = Transformer(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids:  (B, S) — input token IDs.
            kv_caches:  None (prefill) or list of (K,V) per layer (decode).
        Returns:
            logits:     (B, S, vocab_size)
            kv_caches:  updated cache
        """
        B, S = input_ids.shape
        head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads

        # cache_pos: starting position for new tokens in the full sequence.
        # Prefill (kv_caches=None): cache_pos=0. Decode: cache_pos = cache length.
        cache_pos = 0
        if kv_caches is not None:
            cache_pos = kv_caches[0][0].shape[2]

        # Precompute RoPE from 0 to the max position we'll use.
        # Cast to model dtype (bfloat16) to avoid mixed-dtype errors.
        model_dtype = self.model.embed_tokens.weight.dtype
        cos, sin = precompute_freqs_cis(head_dim, cache_pos + S, theta=self.cfg.rope_theta)
        cos = cos.to(device=input_ids.device, dtype=model_dtype)
        sin = sin.to(device=input_ids.device, dtype=model_dtype)

        x, new_caches = self.model(input_ids, cos, sin, cache_pos, kv_caches)
        return self.lm_head(x), new_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive generation with KV-CACHE. Non-streaming: returns all tokens at once."""
        tokens = []
        for token_id in self._generate_impl(
            input_ids, max_new_tokens, temperature, top_p, eos_token_id
        ):
            tokens.append(token_id)
        return torch.cat([input_ids] + tokens, dim=-1)

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ):
        """Autoregressive generation with KV-CACHE. Streaming: yields each token_id as generated."""
        yield from self._generate_impl(
            input_ids, max_new_tokens, temperature, top_p, eos_token_id
        )

    def _generate_impl(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ):
        """
        Shared generation implementation with KV-cache.

        Phase 1 — Prefill: process the full prompt, cache K,V for all layers.
        Phase 2 — Decode:  generate one token at a time, reusing cached K,V.
                           Only 1 new token is processed per step → ~50× faster.
        """
        self.eval()

        # ── Prefill: process full prompt, cache K,V ──────────
        logits, kv_caches = self(input_ids, kv_caches=None)
        logits = logits[:, -1, :]

        next_token = _pick_token(logits, temperature, top_p)
        yield next_token

        # ── Decode: generate token by token using cache ──────
        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            logits, kv_caches = self(next_token, kv_caches=kv_caches)
            logits = logits[:, -1, :]

            next_token = _pick_token(logits, temperature, top_p)
            yield next_token


def _pick_token(
    logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """
    Pick next token: greedy (argmax) when temperature=0, else top-p sampling.
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    return _sample_top_p(logits / temperature, top_p)


def _sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Sample a token from the top-p (nucleus) set."""
    probs = F.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, descending=True)
    probs_cum = torch.cumsum(probs_sort, dim=-1)

    cutoff = probs_cum > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    probs_sort[cutoff] = 0.0

    denom = probs_sort.sum(dim=-1, keepdim=True)
    probs_sort = torch.where(
        denom > 0, probs_sort / denom, torch.zeros_like(probs_sort)
    )
    # Guard: if all probs zero (extreme top_p), fall back to first token
    if (denom == 0).any():
        probs_sort[denom.squeeze(-1) == 0, 0] = 1.0

    next_token = torch.multinomial(probs_sort, 1)
    return probs_idx.gather(-1, next_token)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer and weights from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    cfg = Config.from_hf(MODEL_ID)
    custom_model = SmolLM2ForCausalLM(cfg)
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval().to(device)

    # Weight tying
    custom_model.lm_head.weight = custom_model.model.embed_tokens.weight

    n_params = sum(p.numel() for p in custom_model.parameters())
    print(f"Model loaded: {n_params:,} parameters on {device}")

    import time

    for prompt in ["The capital of France is", "def fibonacci(n):"]:
        print(f"\nPrompt: {prompt}")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        t0 = time.time()
        output_ids = custom_model.generate(
            input_ids, max_new_tokens=20, temperature=0.7, top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        n_new = output_ids.shape[1] - input_ids.shape[1]
        print(f"  {response}")
        print(f"  {n_new} tokens in {elapsed:.1f}s ({n_new/elapsed:.1f} tok/s)")
