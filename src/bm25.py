"""BM25 from scratch: exact-word matching, weighted by rarity and length."""
import math
import re
from collections import Counter, defaultdict

import numpy as np

from src.common import doc_text

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, then split into runs of letters/digits."""
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, corpus: dict) -> "BM25":
        self.doc_ids = list(corpus.keys())
        self.n_docs = len(self.doc_ids)
        # Inverted index: term -> list of (doc position, term frequency)
        self.index = defaultdict(list)
        lengths = np.zeros(self.n_docs)
        for pos, doc_id in enumerate(self.doc_ids):
            tokens = tokenize(doc_text(corpus[doc_id]))
            lengths[pos] = len(tokens)
            for term, tf in Counter(tokens).items():
                self.index[term].append((pos, tf))
        self.avg_len = lengths.mean()
        # length normalisation factor per document, computed once
        self.norm = self.k1 * (1 - self.b + self.b * lengths / self.avg_len)
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.index.items()
        }
        return self

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for this query."""
        s = np.zeros(self.n_docs)
        for term in set(tokenize(query)):
            postings = self.index.get(term)
            if not postings:
                continue  # word never appears in the corpus
            pos = np.fromiter((p for p, _ in postings), dtype=int)
            tf = np.fromiter((t for _, t in postings), dtype=float)
            s[pos] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.norm[pos])
        return s

    def search(self, query: str, k: int = 100) -> list[str]:
        s = self.scores(query)
        k = min(k, self.n_docs)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [self.doc_ids[i] for i in top]