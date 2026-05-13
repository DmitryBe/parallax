"""Parallax throughput benchmark — relevance labeling workload.

Hits /v1/chat/completions in parallel with synthetic (query, candidate) pairs,
measures wall time, tokens/sec, and $/M tokens at the given GPU $/hr.

Usage:
    PARALLAX_URL=https://...modal.run \
    PARALLAX_API_KEY=... \
    python bench/throughput.py --pairs 100 --concurrency 32 --gpu-hourly 1.10
"""
from __future__ import annotations
import argparse
import asyncio
import os
import random
import statistics
import time

import httpx


SYSTEM = (
    "You are a strict relevance judge. Given a search query and a candidate "
    "document, answer with a single word: 'yes' if the candidate is relevant "
    "to the query, 'no' otherwise."
)

# Small synthetic pool — we shuffle/repeat to hit the target count.
POOL = [
    ("dubai marina 2br apartment rent", "2-bedroom apartment in Dubai Marina, 5min walk to promenade, AED 180k/year", True),
    ("python machine learning library", "PyTorch is an open-source deep learning framework developed by Meta AI", True),
    ("best noise-cancelling headphones", "Sony WH-1000XM5 wireless noise cancelling over-ear headphones", True),
    ("iphone 15 pro battery life", "The iPhone 15 Pro Max offers up to 29 hours of video playback", True),
    ("tokyo ramen restaurant recommendation", "Ichiran is famous for tonkotsu ramen with 100+ branches across Japan", True),
    ("how to deploy fastapi on aws", "Tutorial: deploying a FastAPI app to AWS Lambda using Mangum and SAM", True),
    ("vegan dinner recipes quick", "30-minute vegan stir-fry with tofu, broccoli and peanut sauce", True),
    ("london tube map zone 1", "Interactive London Underground map showing all 11 lines and zones 1-9", True),
    # mismatched pairs
    ("dubai marina 2br apartment rent", "Bluetooth speaker, waterproof, 12hr battery, USB-C charging", False),
    ("python machine learning library", "Top 10 sushi restaurants in Osaka", False),
    ("best noise-cancelling headphones", "Recipe: chocolate chip cookies, 30 minutes", False),
    ("iphone 15 pro battery life", "How to grow tomatoes in containers", False),
    ("tokyo ramen restaurant recommendation", "Used cars for sale in Berlin under 5000 EUR", False),
    ("how to deploy fastapi on aws", "Best hiking trails in Patagonia", False),
    ("vegan dinner recipes quick", "Tesla Model 3 review 2025", False),
    ("london tube map zone 1", "Stock market predictions Q3 2025", False),
]


def build_pairs(n: int) -> list[tuple[str, str, bool]]:
    rng = random.Random(42)
    pairs = []
    while len(pairs) < n:
        pairs.append(rng.choice(POOL))
    return pairs[:n]


async def judge_one(client: httpx.AsyncClient, url: str, key: str, model: str, q: str, c: str) -> dict:
    t0 = time.perf_counter()
    r = await client.post(
        f"{url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Query: {q}\nCandidate: {c}\nAnswer:"},
            ],
            "max_tokens": 4,
            "temperature": 0.0,
        },
        timeout=120,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    label_raw = data["choices"][0]["message"]["content"].strip().lower()
    label = label_raw.startswith("y")
    return {
        "latency": dt,
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "label": label,
        "raw": label_raw,
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--model", default=os.environ.get("PARALLAX_MODEL", "qwen2.5-7b"))
    p.add_argument("--gpu-hourly", type=float, default=1.10, help="$/hr for the GPU (A10G≈1.10, L4≈0.80, H100≈3.95)")
    p.add_argument("--openai-input-cost", type=float, default=0.40, help="$/M input tokens for 4.1 mini")
    p.add_argument("--openai-output-cost", type=float, default=1.60, help="$/M output tokens for 4.1 mini")
    args = p.parse_args()

    url = os.environ["PARALLAX_URL"].rstrip("/")
    key = os.environ["PARALLAX_API_KEY"]
    pairs = build_pairs(args.pairs)

    print(f"Parallax @ {url}")
    print(f"Model: {args.model}, pairs: {args.pairs}, concurrency: {args.concurrency}")
    print(f"GPU $/hr: ${args.gpu_hourly:.2f}")
    print()

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    correct = 0

    async with httpx.AsyncClient(http2=False) as client:
        async def worker(q, c, truth):
            async with sem:
                r = await judge_one(client, url, key, args.model, q, c)
                r["truth"] = truth
                r["correct"] = r["label"] == truth
                results.append(r)

        # Warmup: 1 call to ensure engine is hot
        print("warmup...")
        warm = await judge_one(client, url, key, args.model, *POOL[0][:2])
        print(f"  warmup latency: {warm['latency']*1000:.0f}ms  ({warm['prompt_tokens']}→{warm['completion_tokens']} tok)")
        print()

        t0 = time.perf_counter()
        await asyncio.gather(*(worker(q, c, t) for q, c, t in pairs))
        wall = time.perf_counter() - t0

    latencies = sorted(r["latency"] for r in results)
    p50 = latencies[len(latencies)//2]
    p95 = latencies[int(len(latencies)*0.95)]
    p99 = latencies[int(len(latencies)*0.99)]
    total_in = sum(r["prompt_tokens"] for r in results)
    total_out = sum(r["completion_tokens"] for r in results)
    total = total_in + total_out
    tok_per_s = total / wall
    correct = sum(r["correct"] for r in results)

    gpu_cost_per_sec = args.gpu_hourly / 3600.0
    cost_run = gpu_cost_per_sec * wall
    cost_per_M = cost_run / (total / 1_000_000) if total else 0
    # cost-equivalent OpenAI bill for the same token mix
    openai_run = (total_in / 1_000_000) * args.openai_input_cost + (total_out / 1_000_000) * args.openai_output_cost
    openai_per_M = openai_run / (total / 1_000_000) if total else 0

    print(f"=== Results ({args.pairs} pairs, {args.concurrency} concurrent) ===")
    print(f"Wall time:           {wall:.2f}s")
    print(f"Throughput:          {args.pairs/wall:.1f} req/s   {tok_per_s:.0f} tok/s")
    print(f"Latency  p50/p95/p99: {p50*1000:.0f} / {p95*1000:.0f} / {p99*1000:.0f} ms")
    print(f"Tokens:  in={total_in}  out={total_out}  total={total}")
    print(f"Accuracy vs synthetic ground truth: {correct}/{args.pairs} = {correct/args.pairs*100:.1f}%")
    print()
    print(f"--- Cost ---")
    print(f"This run on Parallax @ ${args.gpu_hourly}/hr GPU: ${cost_run:.4f}")
    print(f"  → ${cost_per_M:.3f} per 1M tokens")
    print(f"Same token mix on GPT-4.1 mini API:          ${openai_run:.4f}")
    print(f"  → ${openai_per_M:.3f} per 1M tokens")
    print(f"Savings: {openai_run/max(cost_run,1e-9):.1f}x  ({(1-cost_run/max(openai_run,1e-9))*100:.0f}% cheaper)")


if __name__ == "__main__":
    asyncio.run(main())
