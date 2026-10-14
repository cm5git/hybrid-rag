"""Cross-encoder reranking of a shortlist."""
from src.common import doc_text

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, base, corpus: dict, top_n: int = 50, model_name: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.base = base
        self.corpus = corpus
        self.top_n = top_n
        self.model = CrossEncoder(model_name, max_length=512)

    def search(self, query: str, k: int = 100) -> list[str]:
        candidates = self.base.search(query, max(k, self.top_n))
        head, tail = candidates[: self.top_n], candidates[self.top_n :]
        pairs = [(query, doc_text(self.corpus[d])) for d in head]
        scores = self.model.predict(pairs)  # query and passage read together
        head = [d for _, d in sorted(zip(scores, head), key=lambda x: -x[0])]
        # documents below the shortlist keep their old order, so recall@100 is unchanged
        return (head + tail)[:k]