"""Run at image build time: download the embedding model and precompute corpus embeddings,
so the web app starts in seconds instead of minutes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_corpus  # noqa: E402
from src.dense import DenseRetriever  # noqa: E402

corpus = load_corpus(Path("data/scifact/corpus.jsonl"))
DenseRetriever().fit(corpus)
print(f"Embedded {len(corpus)} documents")