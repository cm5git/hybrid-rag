"""Dense retrieval: embed documents and queries, rank by cosine similarity."""
from pathlib import Path

import numpy as np

from src.common import doc_text

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge models are trained to see this prefix on QUERIES (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str = "data/cache"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        self.cache_dir = Path(cache_dir)

    def fit(self, corpus: dict) -> "DenseRetriever":
        self.doc_ids = list(corpus.keys())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tag = self.model_name.replace("/", "_")
        path = self.cache_dir / f"emb_{tag}_{len(self.doc_ids)}.npy"
        if path.exists():
            self.emb = np.load(path)
        else:
            texts = [doc_text(corpus[i]) for i in self.doc_ids]
            self.emb = self.model.encode(
                texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
            )
            np.save(path, self.emb)
        return self

    def search(self, query: str, k: int = 100) -> list[str]:
        q = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        sims = self.emb @ q  # unit vectors, so dot product = cosine similarity
        k = min(k, len(self.doc_ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [self.doc_ids[i] for i in top]