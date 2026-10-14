"""Score every retrieval system on the same queries, with paired comparisons.

Usage:  python scripts/run_eval.py --systems bm25,dense,hybrid,rerank --limit 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_split  # noqa: E402
from src.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402

METRICS = ["ndcg@10", "mrr@10", "recall@10", "recall@100"]


def per_query_scores(search, queries: dict, qrels: dict) -> dict:
    """One list of scores per metric, aligned by query order."""
    out = {m: [] for m in METRICS}
    for qid, text in queries.items():
        ranked = search(text)
        rel = set(qrels[qid])
        out["ndcg@10"].append(ndcg_at_k(ranked, rel, 10))
        out["mrr@10"].append(reciprocal_rank(ranked[:10], rel))
        out["recall@10"].append(recall_at_k(ranked, rel, 10))
        out["recall@100"].append(recall_at_k(ranked, rel, 100))
    return {m: np.array(v) for m, v in out.items()}


def boot_ci(values, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def paired_diff(a, b, n_boot=10000, seed=42):
    """Mean(b - a) with a CI that resamples the SAME queries for both systems."""
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean(), lo, hi


def build(name, corpus, cache):
    """Build retrievers lazily and reuse them (models are slow to load)."""
    if name in cache:
        return cache[name]
    if name == "bm25":
        from src.bm25 import BM25
        cache[name] = BM25().fit(corpus)
    elif name == "dense":
        from src.dense import DenseRetriever
        cache[name] = DenseRetriever().fit(corpus)
    elif name == "hybrid":
        from src.hybrid import HybridRetriever
        cache[name] = HybridRetriever([build("bm25", corpus, cache), build("dense", corpus, cache)])
    elif name == "rerank":
        from src.rerank import Reranker
        cache[name] = Reranker(build("hybrid", corpus, cache), corpus)
    elif name == "rewrite_hybrid":
        from src.rewrite import QueryRewriter, RewriteThenSearch
        cache[name] = RewriteThenSearch(QueryRewriter(), build("hybrid", corpus, cache))
    else:
        raise SystemExit(f"unknown system: {name}")
    return cache[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--systems", default="bm25,dense,hybrid,rerank")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N queries (0 = all)")
    args = ap.parse_args()

    corpus, queries, qrels = load_split(args.data_dir, args.split)
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])
    print(f"{len(corpus)} documents, {len(queries)} queries ({args.split} split)\n")

    cache, results = {}, {}
    names = args.systems.split(",")
    for name in names:
        retriever = build(name, corpus, cache)
        results[name] = per_query_scores(lambda q: retriever.search(q, 100), queries, qrels)

    print(f"{'system':<16}" + "".join(f"{m:>24}" for m in METRICS))
    for name in names:
        row = f"{name:<16}"
        for m in METRICS:
            v = results[name][m]
            lo, hi = boot_ci(v)
            row += f"{v.mean():>8.3f} [{lo:.3f},{hi:.3f}] "
        print(row)

    print("\nPaired differences in nDCG@10 (later system minus earlier system):")
    for a, b in zip(names, names[1:]):
        mean, lo, hi = paired_diff(results[a]["ndcg@10"], results[b]["ndcg@10"])
        verdict = "real" if lo > 0 or hi < 0 else "could be noise"
        print(f"  {b} - {a}: {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")


if __name__ == "__main__":
    main()