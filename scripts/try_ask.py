import os
import sys

"""Try cited answers on a few questions with the real retriever and the real model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25
from src.data import load_corpus
from src.dense import DenseRetriever
from src.generate import CachedLLM, GeminiLLM, Generator
from src.hybrid import HybridRetriever

QUESTIONS = sys.argv[1:] or [
    "Does aspirin reduce the risk of colorectal cancer?",
    "Who won the 2018 FIFA World Cup?",
    "Does vitamin D supplementation prevent fractures?",
]

corpus = load_corpus(Path("data/scifact/corpus.jsonl"))
retriever = HybridRetriever([BM25().fit(corpus), DenseRetriever().fit(corpus)])
gen = Generator(retriever, corpus, CachedLLM(GeminiLLM()),
                n_passages=int(os.environ.get("N_PASSAGES", 5)))

for q in QUESTIONS:
    a = gen.ask(q)
    print("\nQ:", q)
    print("status:", a.status)
    if a.text:
        print("answer:", a.text)
        print("cites docs:", a.cited)
    if a.note:
        print("note:", a.note)
    print("top passage:", a.passages[0]["title"])