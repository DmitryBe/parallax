# Parallax

**Self-hosted, OpenAI-compatible vLLM endpoints on Modal.**
Bulk inference at GPU-second pricing — for workloads where pay-per-token doesn't make sense (relevance labeling, classification, extraction, enrichment).

## Why

Closed APIs bill per token. For bulk pipelines (millions of (query, candidate) pairs, document enrichment, synthetic data generation) that gets expensive fast. Parallax runs open-weight LLMs on Modal — you pay only for GPU-seconds while the container is hot, and Modal scales to zero between batches.

**Concrete numbers (v1, Qwen2.5-7B on A10G):**

| Workload | Throughput | $/1M tokens | vs GPT-4.1 mini |
|---|---|---|---|
| 100 pairs @ 32 concurrent | 1,654 tok/s | $0.185 | 2.3× cheaper |
| **500 pairs @ 128 concurrent** | **4,643 tok/s** | **$0.066** | **6.5× cheaper** |

Quality A/B on 100 mixed-difficulty (query, candidate) pairs: **100% agreement with Claude Haiku 4.5, Cohen's κ = 1.000.**

## Architecture

One Modal app per model. Each app mounts vLLM's OpenAI-compatible FastAPI server behind a thin Bearer-auth shim, using vLLM's own `build_async_engine_client` lifespan pattern.

```
DSPy / LiteLLM / openai-sdk
        │ base_url=https://<workspace>--parallax-<model>-serve.modal.run/v1
        ▼
Modal ASGI app (FastAPI)
   ├── auth middleware (Bearer PARALLAX_API_KEY)
   └── vLLM OpenAI server mounted at /v1/*
         │
         ▼
   vLLM 0.8.x on @app.function(gpu=...)
   └── weights cached on Modal Volume (cold start ~30s after first run)
```

## Stack

- **Compute:** Modal (scale-to-zero, per-second GPU billing)
- **Server:** vLLM 0.8.x OpenAI-compatible API
- **Auth:** shared Bearer token via Modal Secret
- **Caller:** DSPy + LiteLLM (verified), openai-sdk works too

## Models

Configured in `config/`. v1 ships:
- **`qwen2.5-7b`** → `Qwen/Qwen2.5-7B-Instruct` on A10G 24GB, bf16, `max_model_len=2048`

## Quickstart

```bash
# 1. Generate + store the shared API key
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > .api_key.local
modal secret create parallax-api-key PARALLAX_API_KEY=$(cat .api_key.local)

# 2. Deploy a model
modal deploy parallax/app.py

# 3. Smoke test (DSPy through LiteLLM)
PARALLAX_URL=https://<workspace>--parallax-qwen2-5-7b-serve.modal.run \
PARALLAX_API_KEY=$(cat .api_key.local) \
python tests/test_dspy.py
```

## Adding a new model

1. Add `config/<name>.yaml` (copy `config/qwen2.5-7b.yaml` as a template, adjust `hf_model_id`, `gpu`, `max_model_len`)
2. `MODEL_CONFIG=config/<name>.yaml modal deploy parallax/app.py`

The Modal app name is derived from `name:` in the YAML — separate apps per model = isolated failure domains.

## Calling Parallax

### DSPy

```python
import dspy
lm = dspy.LM(
    model="openai/qwen2.5-7b",
    api_base="https://<workspace>--parallax-qwen2-5-7b-serve.modal.run/v1",
    api_key="<PARALLAX_API_KEY>",
    max_tokens=64,
    temperature=0.0,
)
dspy.configure(lm=lm)

class Relevance(dspy.Signature):
    query: str = dspy.InputField()
    candidate: str = dspy.InputField()
    relevant: bool = dspy.OutputField()

judge = dspy.Predict(Relevance)
out = judge(query="...", candidate="...")
```

### LiteLLM

```python
from litellm import completion
resp = completion(
    model="openai/qwen2.5-7b",
    api_base="https://<workspace>--parallax-qwen2-5-7b-serve.modal.run/v1",
    api_key="<PARALLAX_API_KEY>",
    messages=[{"role": "user", "content": "hi"}],
)
```

## Repo layout

```
parallax/
├── README.md
├── pyproject.toml
├── parallax/
│   ├── app.py          # Modal app + vLLM mount + auth
│   └── config.py       # YAML loader
├── config/
│   └── qwen2.5-7b.yaml
├── tests/
│   └── test_dspy.py    # DSPy smoke tests
├── bench/
│   ├── throughput.py   # tokens/sec + cost/M
│   └── quality_ab.py   # Parallax vs Claude Haiku agreement
└── examples/
    └── dspy_relevance.py
```

## Roadmap

- LiteLLM Proxy in front for unified routing across models
- Additional configs: Qwen2.5-14B, Qwen2.5-32B-AWQ on bigger GPUs
- Optional `/v1/batches` for async large-job submission
- Better cold-start (preload weights, keep-warm strategies)

## License

Apache 2.0
