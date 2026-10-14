"""Hybrid search: fuse several ranked lists with reciprocal rank fusion (RRF)."""
from collections import defaultdict


def rrf(rankings: list[list[str]], k: int = 60, depth: int = 100) -> list[str]:
    """Each list gives a document 1/(k + rank). Sum across lists, sort descending.

    Uses ranks, not scores, because BM25 and cosine scores are on different scales.
    k=60 is the standard default from the original RRF paper; it is not tuned here.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    ordered = sorted(fused, key=lambda d: -fused[d])
    return ordered[:depth]


class HybridRetriever:
    def __init__(self, retrievers: list, k: int = 60, pool: int = 100, depth: int = 100):
        self.retrievers = retrievers
        self.k = k
        self.pool = pool
        self.depth = depth

    def search(self, query: str, k: int | None = None) -> list[str]:
        rankings = [r.search(query, self.pool) for r in self.retrievers]
        return rrf(rankings, k=self.k, depth=k or self.depth)