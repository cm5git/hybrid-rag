"""LLM-as-judge faithfulness check: is every claim in an answer supported by what it cites?

Caveats, stated up front: the judge is an LLM (it can be wrong or lenient), and when it is
the same model family as the generator it may share its blind spots. Treat results on a small
sample as a smoke test, not a benchmark.
"""
JUDGE_PROMPT = """You are checking an answer against the passages it cites.

Passages:
{passages}

Answer:
{answer}

Is every factual claim in the answer supported by the passages above?
Reply with exactly one word: SUPPORTED, PARTIAL (some claims unsupported) or UNSUPPORTED."""

VERDICTS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED")


def parse_verdict(raw: str) -> str:
    """First verdict word found; UNSUPPORTED is checked first because it contains SUPPORTED."""
    up = raw.upper()
    for v in ("UNSUPPORTED", "PARTIAL", "SUPPORTED"):
        if v in up:
            return v
    return "UNPARSEABLE"


def judge(llm, answer) -> str:
    cited = [p for p in answer.passages if p["id"] in set(answer.cited)]
    block = "\n\n".join(f"[{p['n']}] {p['text']}" for p in cited)
    return parse_verdict(llm.complete(JUDGE_PROMPT.format(passages=block, answer=answer.text)))


def summarise(verdicts: list) -> dict:
    counts = {v: verdicts.count(v) for v in VERDICTS + ("UNPARSEABLE",)}
    n = len(verdicts)
    counts["n"] = n
    counts["supported_rate"] = counts["SUPPORTED"] / n if n else 0.0
    return counts