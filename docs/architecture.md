# Architecture

## Design principles

1. **Stay close to vLLM.** vLLM already ships an OpenAI-compatible FastAPI server. We mount it directly rather than building a translation layer. Fewer bugs, less maintenance, free upstream improvements.
2. **One Modal app per model.** Each model gets its own URL, container, GPU, and failure domain. Adding a model never risks breaking another.
3. **Config-driven.** Everything that varies per model lives in a YAML. The `parallax/app.py` is generic.
4. **Auth at the edge.** A single Bearer token via a Modal Secret. No per-user keys, no rate limits — that belongs in a LiteLLM Proxy in front (future).

## Request flow

```
Caller (DSPy / LiteLLM / openai-sdk)
   │  POST /v1/chat/completions
   │  Authorization: Bearer <PARALLAX_API_KEY>
   ▼
Modal edge (HTTPS, auto-scale)
   │
   ▼
@modal.asgi_app() FastAPI process
   ├── BearerAuth middleware
   │     └── public paths: /health, /version, /docs, /openapi.json, /redoc
   ├── CORS middleware
   ├── vLLM's OpenAI router mounted at /v1/*
   │     ├── /v1/chat/completions
   │     ├── /v1/completions
   │     ├── /v1/models
   │     └── /v1/embeddings (when supported)
   │
   ▼
vLLM AsyncLLMEngine (one per container)
   ├── max_num_seqs concurrent sequences
   ├── prefix caching enabled
   └── weights pinned in VRAM
```

## Modal-specific details

| Aspect | How it works |
|---|---|
| **Image** | `modal.Image.debian_slim` + `pip_install("vllm", "fastapi[standard]", "huggingface_hub[hf_transfer]", "pyyaml")` |
| **GPU binding** | `@app.function(gpu="A10G")` — Modal allocates a fresh GPU per container |
| **Weight cache** | `modal.Volume` mounted at `/root/.cache/huggingface` — persists across container restarts; first download is the cold-start tax |
| **Engine cache** | Separate `modal.Volume` at `/root/.cache/vllm` for compiled CUDA graphs / Triton caches |
| **Auth secret** | `modal.Secret.from_name("parallax-api-key", required_keys=["PARALLAX_API_KEY"])` — injected as env var into the container |
| **Concurrency** | `@modal.concurrent(max_inputs=N)` controls how many requests a single container handles in parallel; tune with vLLM's `max_num_seqs` |
| **Scale-to-zero** | `scaledown_window=300` — container shuts down 5 min after last request; `min_containers=0` so cost is zero between batches |
| **Cold start** | ~30s once weights are cached (volume hit), ~90s on first ever start (volume miss + download) |

## The vLLM lifespan quirk (v0.8.x)

vLLM 0.8.x registers its own FastAPI `lifespan` that reads `app.state.log_stats` during startup. If we don't populate `app.state` *before* that lifespan runs, the app crashes with `AttributeError: 'State' object has no attribute 'log_stats'`.

Two patterns we tried that didn't work:
- ❌ `@app.on_event("startup")` runs *after* vLLM's lifespan → too late
- ❌ Building our own engine separately and trying to pass it in → fights upstream

**What works:** override the FastAPI `lifespan_context` with our own that wraps vLLM's `build_async_engine_client`:

```python
from contextlib import asynccontextmanager
from vllm.entrypoints.openai.api_server import (
    build_app, build_async_engine_client, init_app_state,
)

vllm_app = build_app(cli_args)

@asynccontextmanager
async def lifespan(app):
    async with build_async_engine_client(cli_args) as engine_client:
        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, model_config, app.state, cli_args)
        yield

vllm_app.router.lifespan_context = lifespan
```

This is the same pattern vLLM's own `run_server` uses internally — we just substitute our `build_app(...)` result so we can wrap it with middleware afterwards.

## File responsibilities

| File | Role |
|---|---|
| `parallax/app.py` | Modal app definition, image build, the `serve()` ASGI factory, auth middleware |
| `parallax/config.py` | `ModelConfig` dataclass + YAML loader; converts config to vLLM CLI args |
| `config/*.yaml` | Per-model deployment specs (HF model id, GPU, vLLM flags, Modal runtime params) |

## What's intentionally not in v1

- **Multi-model router.** Each model is its own app — callers pick by URL, not by `model=` param across backends. If you ever want unified routing, put a LiteLLM Proxy in front (see [roadmap.md](./roadmap.md)).
- **Streaming SSE.** vLLM supports it natively; we don't disable it, but the labeling workload doesn't need it.
- **Per-user keys / rate limiting.** One shared Bearer token. Multi-tenancy belongs upstream.
- **Async batch API (`/v1/batches`).** Out of scope for v1 — callers handle their own concurrency.
