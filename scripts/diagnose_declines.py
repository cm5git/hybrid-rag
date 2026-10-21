"""For each in-corpus claim in a results file: status vs rank of the best relevant paper.

    python scripts/diagnose_declines.py --split train --results results/answers_train_n5.json
Uses only retrieval and the labels; makes no model calls.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25  # noqa: E402
from src.data import load_split  # noqa: E402
from src.dense import DenseRetriever  # noqa: E402
from src.hybrid import HybridRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--split", default="train")
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    corpus, queries, qrels = load_split(args.data_dir, args.split)
    text_to_qid = {t: q for q, t in queries.items()}
    retriever = HybridRetriever([BM25().fit(corpus), DenseRetriever().fit(corpus)])

    rows = [r for r in json.loads(Path(args.results).read_text()) if r["group"] == "in_corpus"]
    print(f"{'status':<10} {'best relevant rank':<20} claim")
    for r in rows:
        qid = text_to_qid.get(r["question"])
        relevant = set(qrels.get(qid, ()))
        ranked = retriever.search(r["question"], 100)
        ranks = [i for i, d in enumerate(ranked, 1) if d in relevant]
        best = str(ranks[0]) if ranks else ">100"
        print(f"{r['status']:<10} {best:<20} {r['question'][:90]}")


if __name__ == "__main__":
    main()