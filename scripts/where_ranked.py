"""Show where specific documents rank for a question (a retrieval-miss diagnostic)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25
from src.data import load_corpus
from src.dense import DenseRetriever
from src.hybrid import HybridRetriever

QUESTION = "Does vitamin D supplementation prevent fractures?"
WATCH = ["75294", "4462079", "9555784", "11716783", "19878070"]

corpus = load_corpus(Path("data/scifact/corpus.jsonl"))
bm25, dense = BM25().fit(corpus), DenseRetriever().fit(corpus)
systems = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever([bm25, dense])}

for name, system in systems.items():
    ranked = system.search(QUESTION, 100)
    print(f"\n{name}: top 5")
    for i, d in enumerate(ranked[:5], 1):
        print(f"  {i}. {d} | {corpus[d]['title'][:70]}")
    print(f"{name}: rank of the five watched documents")
    for d in WATCH:
        r = ranked.index(d) + 1 if d in ranked else ">100"
        print(f"  {d}: {r}")