"""For each in-corpus claim in the last eval: was a relevant abstract retrieved, and where?

Uses the relevance labels offline, for diagnosis only. No API calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25
from src.data import load_split
from src.dense import DenseRetriever
from src.hybrid import HybridRetriever

corpus, queries, qrels = load_split("data/scifact", "test")
rows = json.loads(Path("results/answers_eval.json").read_text())
text_to_qid = {t: q for q, t in queries.items()}
retriever = HybridRetriever([BM25().fit(corpus), DenseRetriever().fit(corpus)])

for r in rows:
    if r["group"] != "in_corpus":
        continue
    ranked = retriever.search(r["question"], 100)
    relevant = qrels[text_to_qid[r["question"]]]
    ranks = [ranked.index(d) + 1 for d in relevant if d in ranked]
    best = min(ranks) if ranks else ">100"
    print(f"\n[{r['status']}] {r['question'][:120]}")
    print(f"  relevant docs: {len(relevant)}   best relevant rank: {best}")
    print("  top 3: " + " | ".join(corpus[d]["title"][:40] for d in ranked[:3]))