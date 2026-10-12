"""Score any retriever against labelled queries.

A retriever is any function: query text (str) -> ranked list of document IDs (list[str]).
"""
from typing import Callable

import numpy as np

from src.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

Retriever = Callable[[str], list[str]]

METRIC_NAMES = ["ndcg@10", "mrr@10", "recall@10", "recall@100"]


def score_query(ranked_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    """All metrics for a single query."""
    return {
        "ndcg@10": ndcg_at_k(ranked_ids, relevant_ids, 10),
        "mrr@10": reciprocal_rank(ranked_ids[:10], relevant_ids),
        "recall@10": recall_at_k(ranked_ids, relevant_ids, 10),
        "recall@100": recall_at_k(ranked_ids, relevant_ids, 100),
    }


def evaluate(
    retriever: Retriever,
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    depth: int = 100,
) -> dict[str, list[float]]:
    """Run the retriever on every query and return per-query scores for each metric."""
    per_query: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for query_id, text in queries.items():
        ranked_ids = retriever(text)[:depth]
        scores = score_query(ranked_ids, qrels[query_id])
        for name in METRIC_NAMES:
            per_query[name].append(scores[name])
    return per_query


def bootstrap_ci(
    values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def summarise(per_query: dict[str, list[float]], n_boot: int = 1000) -> dict[str, dict[str, float]]:
    """Mean and 95% bootstrap interval for each metric."""
    summary = {}
    for name, values in per_query.items():
        low, high = bootstrap_ci(values, n_boot=n_boot)
        summary[name] = {"mean": float(np.mean(values)), "low": low, "high": high}
    return summary


def format_report(summary: dict[str, dict[str, float]], title: str, n_queries: int) -> str:
    lines = [f"{title}  ({n_queries} queries)"]
    for name, s in summary.items():
        lines.append(f"  {name:<11} {s['mean']:.3f}   95% CI [{s['low']:.3f}, {s['high']:.3f}]")
    return "\n".join(lines)