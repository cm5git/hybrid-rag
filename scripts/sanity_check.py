"""Sanity-check the evaluation harness on the real SciFact data.

A perfect retriever must score ~1.0 and a random one close to 0.
If not, the harness (not the search method) is broken.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_split
from src.evaluate import evaluate, format_report, summarise

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"

corpus, queries, qrels = load_split(DATA_DIR, "test")
text_to_id = {text: qid for qid, text in queries.items()}
all_doc_ids = list(corpus)


def perfect_retriever(query_text):
    relevant = qrels[text_to_id[query_text]]
    others = [d for d in all_doc_ids if d not in relevant]
    return list(relevant) + others


rng = random.Random(0)


def random_retriever(query_text):
    return rng.sample(all_doc_ids, 100)


for name, retriever in [("perfect", perfect_retriever), ("random", random_retriever)]:
    per_query = evaluate(retriever, queries, qrels)
    print(format_report(summarise(per_query), name, len(queries)))
    print()