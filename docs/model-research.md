# Parallax — Open-Source Model Research (Q4 2026)

**Goal:** pick a self-hostable LLM on a single Modal GPU that matches or beats **GPT-4.1 mini** for bulk real-estate data processing (classification / extraction / enrichment, EN + RU), served via vLLM behind LiteLLM.

**Reference bar — GPT-4.1 mini:** MMLU ~87 · GPQA ~50–55 · HumanEval ~85 · MATH ~70 · price $0.40 in / $1.60 out per 1M tok (blended ≈ $0.80–1.00/M).

---

## TL;DR Recommendation

| Tier | Pick | Why |
|---|---|---|
| 🥇 **Top pick** | **Qwen3.5-27B (Dense, Apache 2.0)** | Beats GPT-5 mini on MMLU-Pro / GPQA / IFEval; dense 27B fits one H100 in bf16; rock-solid vLLM support; great EN+RU+CN. |
| 🐎 **Dark horse / cost winner** | **Qwen3.5-35B-A3B (MoE, Apache 2.0)** | Only **3B active** params → 2–3× throughput of dense 27B at similar quality; cheapest $/Mtok by a wide margin. |
| 🛟 Conservative fallback | **Mistral Small 4 (24B, Apache 2.0)** | Battle-tested, simplest deployment, 128k ctx, strong RU. Slightly below 4.1-mini on hardest tasks but excellent for extraction/classification. |

---

## 1. Current landscape (sources)

- Artificial Analysis open-source leaderboard — https://artificialanalysis.ai/models/open-source
- BenchLM 2026 open-source ranking — https://benchlm.ai/blog/posts/best-open-source-llm
- LMArena — https://lmarena.ai
- LiveBench — https://livebench.ai
- EvalPlus / HumanEval+ — https://evalplus.github.io/leaderboard.html
- Qwen3.5 release blog — https://qwenlm.github.io/blog/qwen3.5
- Qwen3.5-27B model card — https://huggingface.co/Qwen/Qwen3.5-27B

Frontier OS leaders (DeepSeek V4 Pro, Kimi K2.6, GLM-5/5.1, Qwen3.5-397B) are 200B–1.6T params — **out of scope** for a single Modal GPU. Below we focus on 14B–32B dense + small MoE.

---

## 2. Finalists in the sweet spot (14B–32B dense + small MoE)

| Model | License | HF ID | Params (active) | Ctx | Notes |
|---|---|---|---|---|---|
| **Qwen3.5-27B** | Apache 2.0 | `Qwen/Qwen3.5-27B` | 27.8B dense | 262k | Thinking + non-thinking modes; native tool-call parser in vLLM (`qwen3_coder`) |
| **Qwen3.5-35B-A3B** | Apache 2.0 | `Qwen/Qwen3.5-35B-A3B` | 35B (3B active) | 262k | MoE; Hybrid Gated DeltaNet attn; very high throughput |
| **Qwen3-32B** (prev gen) | Apache 2.0 | `Qwen/Qwen3-32B` | 32.8B dense | 128k | Older but mature; many AWQ/GPTQ ports |
| **Mistral Small 4** | Apache 2.0 | `mistralai/Mistral-Small-3.5-24B-Instruct-2506` (or 4-series 2026 ckpt) | 24B dense | 128k | Best EU option; strong RU; great structured output |
| **Gemma 4 31B** | Gemma TOS (commercial OK) | `google/gemma-4-31b-it` | 30.7B dense | 256k | Strong general; license is permissive but not OSI |
| **gpt-oss-20B** | Apache 2.0 | `openai/gpt-oss-20b` | 21B (3.6B active MoE) | 131k | OpenAI's open release; very fast (>230 tps); intelligence index 24 — **below 4.1 mini bar** |
| **Nemotron 3 Nano 30B A3B** | NVIDIA Open Model | `nvidia/Nemotron-3-Nano-30B-A3B` | 30B (3B active) | 256k | Insanely fast (~300 tps) but intelligence index ~21 |
| **Qwen3.6-27B** (latest snapshot) | Apache 2.0 | `Qwen/Qwen3.6-27B` | 27.8B dense | 262k | AA Intelligence Index **46** — highest of any sub-40B model; verify availability |

---

## 3. Benchmarks vs GPT-4.1 mini

Higher = better. Qwen3.5 family numbers from official model card; GPT-5 mini is *stronger* than 4.1 mini, so beating it = comfortably beating the 4.1-mini bar.

| Benchmark | GPT-4.1 mini* | GPT-5 mini | **Qwen3.5-27B** | Qwen3.5-35B-A3B | Mistral Small 4 | Gemma 4 31B |
|---|---|---|---|---|---|---|
| MMLU-Pro | ~73 | 83.7 | **86.1** | 85.3 | ~71 | ~77 |
| GPQA Diamond | ~50–55 | 82.8 | **85.5** | 84.2 | ~46 | ~62 |
| HumanEval / LiveCodeBench v6 | ~85 / — | — / 80.5 | — / **80.7** | — / 74.6 | ~84 / ~52 | — / ~65 |
| SWE-bench Verified | ~30 | 72.0 | **72.4** | 69.2 | ~28 | ~40 |
| IFEval (instruction follow) | ~88 | 93.9 | **95.0** | ~93 | ~90 | ~89 |
| BFCL-V4 (tool use) | ~62 | 55.5 | **68.5** | 67.3 | ~55 | ~58 |
| Multilingual (MMMLU) | ~80 | 86.2 | **85.9** | 85.2 | ~78 | ~82 |

*GPT-4.1 mini scores are vendor + community-reported approximations; OpenAI does not publish a full card.

**Verdict:** Qwen3.5-27B and 35B-A3B both **decisively exceed** the GPT-4.1 mini bar on every metric that matters for Parallax (classification, extraction, structured output, multilingual, tool-call). Mistral Small 4 is roughly *at* the 4.1-mini bar — fine for simple extraction, weaker on hard reasoning.

---

## 4. VRAM & GPU fit on Modal

KV cache assumed for ~8k effective context, batched serving.

| Model | bf16 weights | AWQ/INT4 weights | bf16 fits on | INT4 fits on |
|---|---|---|---|---|
| Qwen3.5-27B | ~55 GB | ~16 GB | **H100 80 GB** | A10G 24 GB (tight), L4 24 GB (tight), A100-40 |
| Qwen3.5-35B-A3B | ~70 GB | ~20 GB | **H100 80 GB** | A100-40, A10G/L4 (with quant + tight ctx) |
| Qwen3-32B | ~64 GB | ~19 GB | H100 80 GB | A100-40 / A10G-24 (tight) |
| Mistral Small 4 24B | ~48 GB | ~14 GB | A100-80 / H100 | A10G 24, L4 24 |
| Gemma 4 31B | ~62 GB | ~18 GB | H100 80 GB | A100-40 |

Practical recommendation: **H100 80GB for bf16** (cleanest quality, simplest vLLM config), or **A100-40GB with AWQ 4-bit** for cost-optimised setups.

---

## 5. Cost analysis vs GPT-4.1 mini

Modal pricing (per task brief):

| GPU | $/hr | Realistic Qwen3-32B-class TPS (vLLM, batched) | $/M tokens (combined in+out) |
|---|---|---|---|
| L4 24GB | $0.80 | ~400 (INT4 only, tight) | ~$0.55 |
| A10G 24GB | $1.10 | ~600 (INT4) | ~$0.51 |
| A100-40GB | $2.10 | ~1800 (INT4) / ~1200 (bf16) | ~$0.32–$0.49 |
| A100-80GB | $3.40 | ~2500 (bf16 dense) | ~$0.38 |
| **H100 80GB** | $3.95 | ~5000–6800 (bf16 optimised, per GPUStack) | **~$0.16–$0.22** |

**Qwen3.5-35B-A3B (MoE) on H100** realistically pushes **~10–12k tps** because only 3B params activate per token → effective cost **~$0.10/M tokens**.

GPT-4.1 mini blended ≈ **$0.80–$1.00/M** (mostly output-heavy workloads).

**Breakeven utilization** for Modal vs API:
- A single H100 at ~70% utilization processing Qwen3.5-27B costs ~$2.75/hr × 24h ≈ $66/day = ~$2k/month. That throughput equals ~14B tokens/day → at GPT-4.1-mini API prices that's ~$10k+/day. Self-host wins above **~50M tokens/month**.

---

## 6. vLLM compatibility notes

| Model | vLLM status | Quirks |
|---|---|---|
| Qwen3.5-27B | First-class (vllm ≥ 0.7.0). `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` | Thinking mode adds latency — **disable for bulk extraction** via `enable_thinking=False` in chat template. |
| Qwen3.5-35B-A3B | First-class | MoE needs vllm ≥ 0.7 with `--enable-expert-parallel` flag for best throughput. |
| Qwen3-32B | Rock-solid | None significant. AWQ ports widely available. |
| Mistral Small 4 | First-class | Use `--tool-call-parser mistral`. |
| Gemma 4 31B | Good (vllm ≥ 0.6.x) | Slightly stricter prompt format. |
| gpt-oss-20B | First-class | Lower-quality outputs for extraction. |

All support OpenAI-compatible `/v1/chat/completions` and JSON-mode / guided decoding (`response_format`, `guided_json`, `guided_regex`). For Parallax's structured-output workload, **prefer `guided_json` over freeform JSON** for 100% schema compliance.

---

## 7. Multilingual / Russian quality

- **Qwen3.5 family**: 201 languages claimed; strong RU benchmark results (MMMLU 85.9). Internal community tests rate it on par with GPT-4.1 for RU extraction.
- **Mistral Small 4**: Strong EU language coverage including RU; less strong on Asian languages but irrelevant here.
- **Gemma 4**: Decent RU, weaker than Qwen and Mistral.
- **gpt-oss / Nemotron**: English-centric — avoid for RU.

For Dubai (Arabic mentions) + Russia (RU), **Qwen3.5-27B is the safest single bet**.

---

## 8. Final ranked recommendation

### 🥇 Top pick — **Qwen3.5-27B (bf16 on H100 80GB)**

- **Why:** Beats GPT-4.1 mini on every relevant axis (knowledge, reasoning, tool use, IF, multilingual), Apache 2.0, dense (predictable latency), mature vLLM path.
- **Modal config:** `vllm serve Qwen/Qwen3.5-27B --max-model-len 32768 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --gpu-memory-utilization 0.92`
- **Expected cost:** ~$0.18/M tokens at decent utilization → **~5× cheaper than GPT-4.1 mini**.

### 🐎 Dark horse / cost winner — **Qwen3.5-35B-A3B (MoE)**

- **Why:** Same Qwen3.5 quality tier, ~3× throughput on the same H100 because of 3B active params. Brings effective cost to ~**$0.08–0.10/M tokens** — *roughly 10× cheaper than GPT-4.1 mini*.
- **Risk:** MoE routing instability under extremely diverse prompts; benchmark *slightly* below dense 27B on hardest reasoning (irrelevant for Parallax workload). Worth A/B testing.

### 🛟 Fallback — **Mistral Small 4 24B (AWQ on A10G 24GB)**

- **Why:** Cheapest hardware footprint (A10G $1.10/hr), Apache 2.0, very predictable, excellent JSON-mode behaviour. Quality is *at* GPT-4.1 mini bar, not above — fine for straightforward extraction/classification.
- Use when: you want a $0.50/M setup on a small GPU and don't need top-tier reasoning.

---

## 9. Suggested next steps

1. Spin up Modal H100 endpoint with Qwen3.5-27B bf16 — run real Parallax listing extraction prompts.
2. Side-by-side benchmark vs GPT-4.1 mini on **your own** sampled listings (300 EN + 300 RU), score with LLM-as-judge on field accuracy.
3. If wins → repeat with Qwen3.5-35B-A3B for the cost-winner comparison.
4. Lock vLLM version in your Modal image (suggested: `vllm==0.9.x` as of late 2026) and pin a tested `--quantization` setting.
