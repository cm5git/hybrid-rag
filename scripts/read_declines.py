"""Print claim + relevant abstract for declined claims whose relevant paper was in the passages.

    python scripts/read_declines.py --split train --results results/answers_train_n5.json --max-rank 5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25  # noqa: E402
from src.common import doc_text  # noqa: E402
from src.data import load_split  # noqa: E402
from src.dense import DenseRetriever  # noqa: E402
from src.hybrid import HybridRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--split", default="train")
    ap.add_argument("--results", required=True)
    ap.add_argument("--max-rank", type=int, default=5)
    ap.add_argument("--chars", type=int, default=1500)
    args = ap.parse_args()

    corpus, queries, qrels = load_split(args.data_dir, args.split)
    text_to_qid = {t: q for q, t in queries.items()}
    retriever = HybridRetriever([BM25().fit(corpus), DenseRetriever().fit(corpus)])

    rows = [r for r in json.loads(Path(args.results).read_text())
            if r["group"] == "in_corpus" and r["status"] == "declined"]
    shown = 0
    for r in rows:
        relevant = set(qrels.get(text_to_qid.get(r["question"]), ()))
        for rank, d in enumerate(retriever.search(r["question"], args.max_rank), 1):
            if d in relevant:
                shown += 1
                print("=" * 78)
                print(f"[{shown}] CLAIM: {r['question']}")
                print(f"    relevant paper at rank {rank}, id {d}")
                print(doc_text(corpus[d])[: args.chars])
                break
    print("=" * 78)
    print(f"{shown} declined claims had a relevant paper in the top {args.max_rank}.")


if __name__ == "__main__":
    main()
    