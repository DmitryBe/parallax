"""Minimal DSPy example: relevance labeling against Parallax."""
import os
import dspy

lm = dspy.LM(
    model=f"openai/{os.environ.get('PARALLAX_MODEL', 'qwen3-8b-instruct')}",
    api_base=f"{os.environ['PARALLAX_URL'].rstrip('/')}/v1",
    api_key=os.environ["PARALLAX_API_KEY"],
    max_tokens=64,
    temperature=0.0,
)
dspy.configure(lm=lm)


class Relevance(dspy.Signature):
    """Label whether the candidate is relevant to the query."""
    query: str = dspy.InputField()
    candidate: str = dspy.InputField()
    relevant: bool = dspy.OutputField()


judge = dspy.Predict(Relevance)

pairs = [
    ("dubai marina 2br rent", "2-bedroom apartment in Dubai Marina, AED 180k/year"),
    ("dubai marina 2br rent", "Bluetooth speaker, waterproof, 12hr battery"),
    ("python ml library", "PyTorch is an open-source deep learning framework"),
]

for q, c in pairs:
    out = judge(query=q, candidate=c)
    print(f"{out.relevant!s:5}  q={q!r}  c={c[:50]!r}")
