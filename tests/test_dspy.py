"""DSPy smoke test against a deployed Parallax endpoint.

Usage:
    PARALLAX_URL=https://<workspace>--parallax-qwen3-8b.modal.run \
    PARALLAX_API_KEY=<key> \
    python tests/test_dspy.py
"""
from __future__ import annotations
import os
import sys

import dspy


def main() -> int:
    base_url = os.environ.get("PARALLAX_URL", "").rstrip("/")
    api_key = os.environ.get("PARALLAX_API_KEY", "")
    model_name = os.environ.get("PARALLAX_MODEL", "qwen3-8b-instruct")

    if not base_url or not api_key:
        print("ERROR: set PARALLAX_URL and PARALLAX_API_KEY")
        return 2

    lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=f"{base_url}/v1",
        api_key=api_key,
        max_tokens=64,
        temperature=0.0,
    )
    dspy.configure(lm=lm)

    # --- Test 1: raw completion ---------------------------------------------
    print("[1/3] raw LM call...")
    out = lm("Say the single word: hello")
    print(f"      -> {out!r}")

    # --- Test 2: typed signature (the labeling shape we actually care about) -
    print("[2/3] typed signature — relevance label...")

    class Relevance(dspy.Signature):
        """Decide if the candidate document is relevant to the search query."""
        query: str = dspy.InputField()
        candidate: str = dspy.InputField()
        relevant: bool = dspy.OutputField(desc="True if candidate answers the query")

    judge = dspy.Predict(Relevance)
    r1 = judge(
        query="best 2-bedroom apartment near marina",
        candidate="Spacious 2BR apartment in Dubai Marina, 5min walk to the promenade.",
    )
    r2 = judge(
        query="best 2-bedroom apartment near marina",
        candidate="Wireless earbuds, noise cancelling, 30hr battery.",
    )
    print(f"      relevant pair -> {r1.relevant}")
    print(f"      irrelevant pair -> {r2.relevant}")
    assert r1.relevant is True, "expected relevant=True for marina pair"
    assert r2.relevant is False, "expected relevant=False for earbuds pair"

    # --- Test 3: graded score with rationale --------------------------------
    print("[3/3] graded score signature...")

    class GradedRelevance(dspy.Signature):
        """Rate how relevant the candidate is to the query on a 0-3 TREC scale."""
        query: str = dspy.InputField()
        candidate: str = dspy.InputField()
        score: int = dspy.OutputField(desc="0=not relevant, 1=marginal, 2=relevant, 3=highly relevant")

    graded = dspy.Predict(GradedRelevance)
    g = graded(
        query="machine learning frameworks",
        candidate="PyTorch is a popular open-source ML framework developed by Meta.",
    )
    print(f"      score -> {g.score}")
    assert 0 <= int(g.score) <= 3

    print("\nAll DSPy smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
