"""Retrieval metrics.

Every function takes:
    ranked_ids   - document IDs in the order the system returned them (best first)
    relevant_ids - the set of IDs that are correct for the query
"""
import math


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = len(set(top_k) & relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant documents that appear in the top k."""
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits = len(set(top_k) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant document, or 0.0 if none is returned."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not relevant_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Best possible ordering: all relevant documents at the top (capped at k).
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg