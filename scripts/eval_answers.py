"""Smoke-test answer generation: citation/decline behaviour and faithfulness.

Two groups of questions:
  off_topic  : hand-written questions unrelated to biomedical abstracts (should be declined)
  in_corpus  : SciFact claims from the chosen split (passages should hold relevant evidence)

Tune on the TRAIN split; look at TEST once, at the end. Results are cached on disk and the
run stops cleanly when the call budget is spent, so you can re-run to continue.

Examples:
    python scripts/eval_answers.py --split train --n-passages 5
    python scripts/eval_answers.py --split train --n-passages 8
    python scripts/eval_answers.py --split test --n-passages 5 --offset 15
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.faithfulness import judge, summarise  # noqa: E402
from src.generate import CachedLLM, GeminiLLM, Generator, LLMUnavailable  # noqa: E402

OFF_TOPIC = [
    "Who won the 2018 FIFA World Cup?",
    "What is the capital of Australia?",
    "How do I bake sourdough bread?",
    "What is the best programming language for web development?",
    "How many moons does Jupiter have?",
    "Who wrote the novel Pride and Prejudice?",
    "What is the exchange rate between pounds and euros?",
    "How do electric guitars produce sound?",
]


class Budget:
    """Counts REAL (uncached) model calls, paces them for the free tier (15 per minute),
    and raises when the budget for this run is spent."""

    def __init__(self, llm: CachedLLM, max_calls: int):
        self.llm, self.max_calls, self.start = llm, max_calls, llm.calls

    def complete(self, prompt):
        cached = hashlib.sha256(prompt.encode()).hexdigest() in self.llm.cache
        if not cached:
            if self.llm.calls - self.start >= self.max_calls:
                raise LLMUnavailable("call budget for this run is spent")
            time.sleep(4.5)
        return self.llm.complete(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--n-claims", type=int, default=15)
    ap.add_argument("--offset", type=int, default=0, help="skip this many claims first")
    ap.add_argument("--n-passages", type=int, default=5)
    ap.add_argument("--max-calls", type=int, default=40, help="real API calls this run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/answers_{args.split}_n{args.n_passages}.json"

    from src.bm25 import BM25
    from src.data import load_split
    from src.dense import DenseRetriever
    from src.hybrid import HybridRetriever

    corpus, queries, _ = load_split(args.data_dir, args.split)
    claims = list(queries.values())[args.offset: args.offset + args.n_claims]
    retriever = HybridRetriever([BM25().fit(corpus), DenseRetriever().fit(corpus)])
    llm = Budget(CachedLLM(GeminiLLM()), args.max_calls)
    gen = Generator(retriever, corpus, llm, n_passages=args.n_passages)

    rows, stopped = [], False
    for group, items in (("off_topic", OFF_TOPIC), ("in_corpus", claims)):
        for q in items:
            try:
                a = gen.ask(q)
                if a.status == "retrieval_only":
                    raise LLMUnavailable(a.note)
                verdict = judge(llm, a) if a.status == "answered" else None
            except LLMUnavailable as exc:
                print(f"Stopping: {exc}. Run again later; finished answers are cached.")
                stopped = True
                break
            rows.append({"group": group, "question": q, "status": a.status,
                         "answer": a.text, "cited": a.cited, "verdict": verdict})
        if stopped:
            break

    def statuses(g):
        return [r["status"] for r in rows if r["group"] == g]

    print(f"\nsplit={args.split} n_passages={args.n_passages} offset={args.offset}")
    print(f"{len(rows)} questions finished ({'stopped early' if stopped else 'complete'})")
    for g in ("off_topic", "in_corpus"):
        s = statuses(g)
        print(f"  {g:<10} n={len(s):<3} answered={s.count('answered')} "
              f"declined={s.count('declined')} withheld(uncited)={s.count('uncited')}")
    verdicts = [r["verdict"] for r in rows if r["verdict"]]
    if verdicts:
        print("  faithfulness (LLM judge, answered only):", summarise(verdicts))
    for r in rows:
        if r["group"] == "off_topic" and r["status"] == "answered":
            print("  off-topic question that was ANSWERED:", r["question"])

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=1))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()