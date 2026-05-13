"""Quality A/B: Parallax (Qwen2.5-7B) vs Claude Haiku on the same labeling pairs.

Strategy: Claude Haiku is the strong-baseline judge. We measure how often
Parallax (the cheap self-hosted model) agrees with Haiku's labels.
"""
from __future__ import annotations
import asyncio
import json
import os
import time

import httpx

SYSTEM = (
    "You are a strict relevance judge. Given a search query and a candidate "
    "document, answer with a single word: 'yes' if the candidate is relevant "
    "to the query, 'no' otherwise."
)

# Larger pool than the throughput bench — varied difficulty
POOL = [
    # clear positives
    ("dubai marina 2br apartment rent", "2-bedroom apartment in Dubai Marina, 5min to promenade, AED 180k/year"),
    ("python machine learning library", "PyTorch is an open-source deep learning framework developed by Meta AI"),
    ("best noise-cancelling headphones", "Sony WH-1000XM5 wireless noise cancelling over-ear headphones"),
    ("iphone 15 pro battery life", "The iPhone 15 Pro Max offers up to 29 hours of video playback"),
    ("tokyo ramen restaurant recommendation", "Ichiran is famous for tonkotsu ramen with 100+ branches in Japan"),
    ("how to deploy fastapi on aws", "Tutorial: deploying a FastAPI app to AWS Lambda using Mangum and SAM"),
    ("vegan dinner recipes quick", "30-minute vegan stir-fry with tofu, broccoli and peanut sauce"),
    ("london tube map zone 1", "Interactive London Underground map showing all 11 lines and zones 1-9"),
    # clear negatives
    ("dubai marina 2br apartment rent", "Bluetooth speaker, waterproof, 12hr battery, USB-C charging"),
    ("python machine learning library", "Top 10 sushi restaurants in Osaka"),
    ("best noise-cancelling headphones", "Recipe: chocolate chip cookies, 30 minutes"),
    ("iphone 15 pro battery life", "How to grow tomatoes in containers"),
    ("tokyo ramen restaurant recommendation", "Used cars for sale in Berlin under 5000 EUR"),
    ("how to deploy fastapi on aws", "Best hiking trails in Patagonia"),
    ("vegan dinner recipes quick", "Tesla Model 3 review 2025"),
    ("london tube map zone 1", "Stock market predictions Q3 2025"),
    # ambiguous / harder
    ("affordable family car", "The Toyota Corolla offers great fuel efficiency and reliability"),
    ("affordable family car", "Ferrari 488 GTB priced at $250k"),
    ("learn javascript for beginners", "Eloquent JavaScript: A modern introduction to programming"),
    ("learn javascript for beginners", "Advanced compiler optimization in LLVM"),
    ("best italian restaurant in NYC", "Carbone in Greenwich Village serves classic Italian-American cuisine"),
    ("best italian restaurant in NYC", "McDonald's launches new chicken sandwich"),
    ("react vs vue framework", "Comparison of React.js and Vue.js for modern web applications"),
    ("react vs vue framework", "How to bake sourdough bread at home"),
    ("climate change effects on agriculture", "Rising temperatures reduce wheat yields across Europe"),
    ("climate change effects on agriculture", "New iPhone case design released"),
    ("how to invest in index funds", "Vanguard S&P 500 ETF: low-cost passive investment guide"),
    ("how to invest in index funds", "Top 5 anime of summer 2024"),
    # tricky — semantically related but not directly relevant
    ("python web scraping tutorial", "BeautifulSoup is a Python library for parsing HTML and XML"),  # relevant
    ("python web scraping tutorial", "Python is named after Monty Python comedy group"),  # not relevant
    ("dubai marina 2br apartment rent", "Studio apartment for sale in JBR, Dubai, AED 1.2M"),  # marginal
    ("dubai marina 2br apartment rent", "Dubai weather forecast for next week"),  # not relevant
]


def build_pairs(n: int) -> list[tuple[str, str]]:
    import random
    rng = random.Random(42)
    out = []
    while len(out) < n:
        out.append(rng.choice(POOL))
    return out[:n]


async def judge_parallax(client, url, key, model, q, c) -> bool:
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
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip().lower()
    return txt.startswith("y")


async def judge_claude(client, key, q, c) -> bool:
    r = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 8,
            "system": SYSTEM,
            "messages": [
                {"role": "user", "content": f"Query: {q}\nCandidate: {c}\nAnswer:"},
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    txt = r.json()["content"][0]["text"].strip().lower()
    return txt.startswith("y")


async def main():
    n = int(os.environ.get("N_PAIRS", "100"))
    parallax_url = os.environ["PARALLAX_URL"].rstrip("/")
    parallax_key = os.environ["PARALLAX_API_KEY"]
    parallax_model = os.environ.get("PARALLAX_MODEL", "qwen2.5-7b")
    claude_key = os.environ["ANTHROPIC_API_KEY"]

    pairs = build_pairs(n)
    print(f"A/B: Parallax ({parallax_model}) vs Claude Haiku, {n} pairs")
    print()

    sem_p = asyncio.Semaphore(32)
    sem_c = asyncio.Semaphore(8)  # be polite to Anthropic
    parallax_labels: list[bool | None] = [None] * n
    claude_labels: list[bool | None] = [None] * n

    async with httpx.AsyncClient() as client:
        async def run_p(i, q, c):
            async with sem_p:
                parallax_labels[i] = await judge_parallax(client, parallax_url, parallax_key, parallax_model, q, c)

        async def run_c(i, q, c):
            async with sem_c:
                try:
                    claude_labels[i] = await judge_claude(client, claude_key, q, c)
                except httpx.HTTPStatusError as e:
                    print(f"Claude error pair {i}: {e.response.status_code} {e.response.text[:200]}")
                    raise

        t0 = time.perf_counter()
        await asyncio.gather(
            *(run_p(i, q, c) for i, (q, c) in enumerate(pairs)),
            *(run_c(i, q, c) for i, (q, c) in enumerate(pairs)),
        )
        wall = time.perf_counter() - t0

    # Compare
    agree = sum(1 for p, c in zip(parallax_labels, claude_labels) if p == c)
    p_yes = sum(1 for x in parallax_labels if x)
    c_yes = sum(1 for x in claude_labels if x)
    disagreements = [(i, pairs[i], parallax_labels[i], claude_labels[i])
                     for i in range(n) if parallax_labels[i] != claude_labels[i]]

    print(f"Wall time:              {wall:.1f}s")
    print(f"Parallax  'yes' count:  {p_yes}/{n}")
    print(f"Claude    'yes' count:  {c_yes}/{n}")
    print(f"Agreement:              {agree}/{n} = {agree/n*100:.1f}%")
    print(f"Disagreements:          {len(disagreements)}")
    print()
    if disagreements:
        print("=== Disagreements (Parallax / Claude) ===")
        for i, (q, c), pl, cl in disagreements[:20]:
            print(f"  [{i}] P={pl!s:5} C={cl!s:5}  q={q!r}")
            print(f"        candidate={c[:90]!r}")

    # Cohen's kappa
    def kappa(a, b):
        n = len(a)
        po = sum(1 for x, y in zip(a, b) if x == y) / n
        pa = sum(1 for x in a if x) / n
        pb = sum(1 for x in b if x) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        return (po - pe) / (1 - pe) if pe < 1 else 1.0

    k = kappa(parallax_labels, claude_labels)
    print()
    print(f"Cohen's kappa:          {k:.3f}")
    print(f"  (0.81-1.0 = almost perfect, 0.61-0.80 = substantial, 0.41-0.60 = moderate)")


if __name__ == "__main__":
    asyncio.run(main())
