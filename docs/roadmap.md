# Roadmap

What's shipped, what's next, what's deliberately out of scope.

## v1 — shipped

- ✅ OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- ✅ Qwen2.5-7B-Instruct on A10G via Modal
- ✅ Bearer-token auth via Modal Secret
- ✅ Modal Volume for weight + engine caches (warm cold starts)
- ✅ YAML-driven model config (one app per model)
- ✅ DSPy + LiteLLM verified end-to-end
- ✅ Throughput + quality benchmarks documented

## v1.1 — small wins (next 1-2 days work)

- **Push to GitHub** — set up the `DmitryBe/parallax` remote, add CI for `tests/test_dspy.py` against a deployed endpoint (smoke check on PR)
- **Fix `/version` route shadowing** — vLLM registers its own `/version` that takes precedence over ours; rename to `/parallax/version` or register before `build_app`
- **Health check endpoint** — return engine status, current `max_num_seqs` utilization, queue depth (useful for caller-side adaptive concurrency)
- **Cost-aware logging** — log per-request token counts and accumulated $ to Modal's logs (so cost shows up in dashboards without extra plumbing)

## v1.2 — second model

Goal: prove the "add a model = add a YAML" loop with a meaningfully different model.

Candidates:

| Model | GPU | Why |
|---|---|---|
| **Qwen2.5-14B-Instruct-AWQ** | A10G 24GB | Stronger labeling for hard pairs; 4-bit fits same GPU |
| **Qwen2.5-7B-Instruct-AWQ** | L4 16GB | Cheapest setup; benchmarks vs 7B-bf16 on quality/cost |
| **Mistral-Small-2501-AWQ (24B)** | L4 or A10G | Different family for diversity |
| **Qwen3-30B-A3B (MoE)** | H100 80GB | Cost winner candidate (3B active params) — if vLLM supports it |

Whichever is chosen, add: `config/<model>.yaml`, run benchmarks, document deltas.

## v2 — LiteLLM Proxy front door

When 3+ models are running and tracking N URLs becomes annoying, put a LiteLLM Proxy in front:

```
DSPy ──► LiteLLM Proxy ──► parallax-qwen2.5-7b
                       ├──► parallax-qwen2.5-14b
                       └──► parallax-mistral-small
```

LiteLLM Proxy is an existing tool — we don't build it. We add:
- `proxy/config.yaml` listing all Parallax backends as OpenAI-compatible upstreams
- A README section on running the proxy (locally or on Modal itself)
- Optional: cost tracking dashboard via LiteLLM's built-in spend logging

## v3 — async batch API

For workloads where the caller wants to fire off 1M pairs and poll for completion, add `/v1/batches` (OpenAI Batch API shape). Possible implementations:

- Implement vLLM-side via existing OpenAI-compatible batch support if it ships
- Or build a thin Modal-native job queue: caller POSTs a JSONL of requests, gets a job ID, polls for `/v1/batches/{id}` → returns results JSONL when done

Real win: lets the caller fully decouple from connection lifecycle, and we can pack the batch across container scaleups.

## Out of scope (deliberately)

- **Per-user API keys / multi-tenancy.** Build with LiteLLM Proxy + key rotation instead.
- **Fine-tuning.** Use a separate Modal app for TRL/Axolotl/Unsloth; Parallax serves inference only.
- **Tool calling / function calling.** vLLM gives us whatever it gives us — no Parallax-specific surface.
- **Streaming UI.** Bulk labeling doesn't need it; the bare endpoint supports SSE for callers who do.
- **Local-first / non-Modal deploy.** Modal is the value prop (scale-to-zero, GPU billing). If you want bare metal, use vLLM directly.

## Open research questions

These would feed back into roadmap decisions:

1. **What's the quality/cost frontier?** Run a calibration study: Qwen2.5-{4B, 7B, 14B, 32B-AWQ} × {bf16, AWQ-4bit} × {A10G, L4, H100} on a real held-out eval set with human gold labels. Currently we only have synthetic agreement vs Claude — not enough signal.
2. **Where does prefix caching break down?** Workloads with rotating system prompts (e.g. dynamic few-shot) lose the caching benefit. Profile this on a realistic workload.
3. **Does AWQ 4-bit hurt labeling accuracy?** Quantization is "free" on benchmarks, but real workloads sometimes regress. Need an A/B between Qwen2.5-7B bf16 vs Qwen2.5-7B-AWQ on real pairs.
4. **What's the breakeven volume vs OpenAI Batch API?** OpenAI offers 50% off via Batch API; their effective cost at the workload's token mix is closer to ~$0.21/M. Below some volume threshold, just using their Batch API is simpler.
