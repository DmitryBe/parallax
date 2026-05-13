# Benchmarks

All numbers from real runs against the deployed `parallax-qwen2.5-7b` endpoint on Modal A10G. Reproduce with [`bench/throughput.py`](../bench/throughput.py) and [`bench/quality_ab.py`](../bench/quality_ab.py).

## Workload

**Task:** binary relevance labeling. Given a `(query, candidate)` pair, the model returns `yes` or `no`.

**Prompt shape:**
- System: ~30 token relevance-judge instruction (shared across all pairs → benefits from prefix caching)
- User: `Query: {query}\nCandidate: {candidate}\nAnswer:`
- Output: 1-4 tokens (`yes` / `no` plus possible padding)

**Token distribution:** input-heavy (~80 input : 2 output per pair). This shape favors prefix-caching-aware backends like vLLM but is *less* favorable to per-token cost stories than a balanced workload (closed APIs charge less for input than output).

## Throughput

| Run | Pairs | Concurrency | Wall time | Throughput | $/1M tokens (A10G @ $1.10/hr) |
|---|---|---|---|---|---|
| Small | 100 | 32 | 4.98s | 20.1 req/s · 1,654 tok/s | $0.185 |
| **Production** | **500** | **128** | **8.82s** | **56.7 req/s · 4,643 tok/s** | **$0.066** |

**Latency at production concurrency (128):** p50 2.0s · p95 2.8s · p99 3.5s.

The 2.8× throughput jump from concurrency 32→128 confirms the A10G + Qwen-7B combo wasn't saturated at 32. Beyond 128 there'd be diminishing returns until input length grows.

## Cost comparison

For 1M tokens at the labeling workload's input/output ratio:

| Provider | $/1M tokens (effective) | Notes |
|---|---|---|
| **Parallax (Qwen2.5-7B, A10G, 128 concurrency)** | **$0.066** | self-hosted, steady state |
| Parallax (same, 32 concurrency) | $0.185 | unsaturated GPU |
| GPT-4.1 mini | $0.429 | at the same input/output token mix |
| GPT-4o-mini | ~$0.30 | for reference |
| Claude Haiku 4.5 | ~$0.50 | for reference |

**Headline:** Parallax is **6.5× cheaper** than GPT-4.1 mini on this workload at production concurrency.

**Caveat:** the "$0.066/M tokens" number is workload-dependent. Workloads with more output tokens (generation-heavy) widen the gap further (open-weight has no input/output price asymmetry); workloads dominated by short prompts narrow it.

## Quality A/B

Comparing Parallax's labels against Claude Haiku 4.5 as a strong baseline judge. Same 100 pairs, both at `temperature=0`.

**Pair composition (32 unique templates, 100 sampled):**
- 16 clear positives + negatives (real estate, ML libraries, restaurants, travel)
- 12 mid-difficulty (ambiguous semantic relevance)
- 4 tricky (lexically related but semantically off-topic)

| Metric | Value |
|---|---|
| Parallax "yes" count | 45/100 |
| Claude Haiku "yes" count | 45/100 |
| **Agreement** | **100/100 (100%)** |
| **Cohen's κ** | **1.000** |
| Disagreements | 0 |

### What this means

Strong positive signal that Qwen2.5-7B handles **clear-cut** relevance labeling at frontier-mini quality. The κ=1.0 result is suspiciously clean — likely the pair pool is too easy. Real eval should use:

1. **Graded scoring (0-3 TREC scale)** instead of binary — exposes calibration differences
2. **Genuinely ambiguous pairs** (partial relevance, near-duplicates, multi-aspect queries)
3. **Your actual workload pairs** with human gold labels for ~200-500 samples

Defer harder quality evaluation until the downstream `llmjudge` pipeline produces real workload data.

## Reproducing

```bash
# Throughput
PARALLAX_URL=https://<workspace>--parallax-qwen2-5-7b-serve.modal.run \
PARALLAX_API_KEY=$(cat .api_key.local) \
python bench/throughput.py --pairs 500 --concurrency 128 --gpu-hourly 1.10

# Quality A/B (requires ANTHROPIC_API_KEY in env)
PARALLAX_URL=https://<workspace>--parallax-qwen2-5-7b-serve.modal.run \
PARALLAX_API_KEY=$(cat .api_key.local) \
ANTHROPIC_API_KEY=sk-ant-... \
python bench/quality_ab.py
```

## Open questions for future benchmarks

- How does throughput scale at `max_num_seqs=256` (the configured ceiling) vs 128?
- How does AWQ 4-bit quantization affect quality vs cost for Qwen2.5-7B?
- Does Qwen2.5-14B significantly beat 7B on hard pairs, justifying L4 ($0.80/hr) or A100-40 ($2.10/hr)?
- What's the break-even concurrency vs latency tradeoff for different downstream pipelines?
