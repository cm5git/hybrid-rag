"""Entry point: loads SciFact, builds BM25 + embeddings + hybrid fusion, serves the API.

The reranker is left out on purpose: on SciFact it did not improve nDCG@10 (see README).
"""
import os
from pathlib import Path

from app.main import create_app
from src.bm25 import BM25
from src.data import load_corpus
from src.dense import DenseRetriever
from src.generate import CachedLLM, GeminiLLM, Generator, LLMUnavailable
from src.hybrid import HybridRetriever

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/scifact"))


def build_generator() -> Generator:
    corpus = load_corpus(DATA_DIR / "corpus.jsonl")
    retriever = HybridRetriever([BM25().fit(corpus), DenseRetriever().fit(corpus)])
    try:
        llm = CachedLLM(GeminiLLM())
    except LLMUnavailable:
        llm = None  # no API key: the demo still works as a search engine
    return Generator(retriever, corpus, llm)


app = create_app(build_generator())