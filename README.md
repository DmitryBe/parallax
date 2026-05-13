# Parallax

> **Self-hosted, OpenAI-compatible LLM endpoints on Modal — for bulk inference where pay-per-token doesn't make sense.**

Parallax exposes any open-weight LLM (Qwen, Mistral, Llama, etc.) as an OpenAI-compatible API. It runs on Modal's per-second GPU billing, scales to zero between batches, and is callable from any client that speaks OpenAI's protocol — DSPy, LiteLLM, openai-sdk, you name it.

**Use it when:** you're processing millions of (query, candidate) pairs, classifying documents, extracting structured data, generating synthetic training labels — workloads where token-based pricing on closed APIs becomes prohibitive.

**Don't use it when:** you have low volume (<10k requests/day), need very low latency (<200ms), or need frontier-model quality (GPT-5 / Claude Opus 4.5).

---

## What's in it

- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings` (when supported by the model). No translation layer.
- **One Modal app per model** — clean isolation, no cross-model failure blast radius. Add a model = add a YAML + `modal deploy`.
- **vLLM-powered** — high throughput, prefix caching, guided decoding (JSON schema, regex), AWQ/GPTQ quantization support.
- **Bearer-token auth** via a single Modal Secret. Simple, sufficient for trusted-caller setups.
- **Scale-to-zero** — Modal spins containers down after `scaledown_window` seconds idle. You pay $0 between batches.

## Quickstart

```bash
git clone https://github.com/DmitryBe/parallax.git
cd parallax

# 1. Generate + store the API key
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > .api_key.local
modal secret create parallax-api-key PARALLAX_API_KEY=$(cat .api_key.local)

# 2. Deploy the default model (Qwen2.5-7B on A10G)
modal deploy parallax/app.py

# 3. Smoke test through DSPy
PARALLAX_URL=https://<workspace>--parallax-qwen2-5-7b-serve.modal.run \
PARALLAX_API_KEY=$(cat .api_key.local) \
python tests/test_dspy.py
```

Expected output:
```
[1/3] raw LM call...                 -> ['hello']
[2/3] typed signature — relevance... -> True / False
[3/3] graded score signature...      -> 2
All DSPy smoke tests passed.
```

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
    """Decide if the candidate is relevant to the query."""
    query: str = dspy.InputField()
    candidate: str = dspy.InputField()
    relevant: bool = dspy.OutputField()

judge = dspy.Predict(Relevance)
out = judge(query="...", candidate="...")
print(out.relevant)  # True / False
```

### LiteLLM

```python
from litellm import completion

resp = completion(
    model="openai/qwen2.5-7b",
    api_base="https://<workspace>--parallax-qwen2-5-7b-serve.modal.run/v1",
    api_key="<PARALLAX_API_KEY>",
    messages=[{"role": "user", "content": "label this"}],
)
```

### Raw HTTP

```bash
curl -H "Authorization: Bearer <KEY>" -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"hi"}],"max_tokens":16}' \
  https://<workspace>--parallax-qwen2-5-7b-serve.modal.run/v1/chat/completions
```

## Models

Configured in [`config/`](./config). Currently shipping:

| Name | HF model | GPU | Context | Notes |
|---|---|---|---|---|
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | A10G 24GB bf16 | 2048 | v1 default — strong multilingual, Apache 2.0 |

**Adding a new model:** copy `config/qwen2.5-7b.yaml`, edit `hf_model_id`, `gpu`, `max_model_len`, then `MODEL_CONFIG=config/<your-config>.yaml modal deploy parallax/app.py`.

## Performance & cost

Measured on the relevance-labeling workload (80 tok input + 2 tok output per pair, system prompt shared):

| Run | Concurrency | Throughput | $/1M tokens | vs GPT-4.1 mini |
|---|---|---|---|---|
| 100 pairs | 32 | 1,654 tok/s | $0.185 | 2.3× cheaper |
| **500 pairs** | **128** | **4,643 tok/s** | **$0.066** | **6.5× cheaper** |

Quality: **100% agreement with Claude Haiku 4.5** on a 100-pair mixed-difficulty labeling set (Cohen's κ = 1.0). See [`docs/benchmarks.md`](./docs/benchmarks.md) for the methodology and full breakdown.

## Project docs

- **[`docs/architecture.md`](./docs/architecture.md)** — how Parallax is structured and the deploy model
- **[`docs/benchmarks.md`](./docs/benchmarks.md)** — throughput + quality results in full
- **[`docs/model-research.md`](./docs/model-research.md)** — open-source model landscape vs GPT-4.1 mini (research that informed model selection)
- **[`docs/operations.md`](./docs/operations.md)** — deploy, redeploy, debug, costs, ops runbook
- **[`docs/roadmap.md`](./docs/roadmap.md)** — what's next: more models, LiteLLM Proxy, batch API

## Repo layout

```
parallax/
├── README.md                   # this file
├── pyproject.toml
├── parallax/                   # the Modal app
│   ├── app.py                  # Modal entrypoint + vLLM mount + auth shim
│   ├── config.py               # YAML config loader
│   └── __init__.py
├── config/                     # one YAML per model deployment
│   └── qwen2.5-7b.yaml
├── tests/
│   └── test_dspy.py            # DSPy + LiteLLM smoke tests
├── bench/
│   ├── throughput.py           # tokens/sec + $/M cost benchmark
│   └── quality_ab.py           # Parallax vs Claude Haiku agreement
├── examples/
│   └── dspy_relevance.py       # minimal DSPy labeling example
└── docs/                       # see "Project docs" above
```

## Requirements

- Modal account ([modal.com](https://modal.com)) with GPU quota (A10G minimum; H100 for 30B+ models)
- Python 3.11+
- ~$0.50 in Modal credit to run all the benchmarks

## License

Apache 2.0
