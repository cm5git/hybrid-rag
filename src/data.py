"""Loaders for BEIR-format datasets (corpus.jsonl, queries.jsonl, qrels/<split>.tsv)."""
import csv
import json
from pathlib import Path


def load_corpus(path: Path) -> dict[str, dict[str, str]]:
    """Return {doc_id: {"title": ..., "text": ...}}."""
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = {
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            }
    return corpus


def load_queries(path: Path) -> dict[str, str]:
    """Return {query_id: query_text} for every query in the file."""
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    return queries


def load_qrels(path: Path) -> dict[str, set[str]]:
    """Return {query_id: {relevant_doc_id, ...}}, keeping only positive scores."""
    qrels: dict[str, set[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    return qrels


def load_split(data_dir: Path, split: str = "test"):
    """Load corpus, the queries that have labels in `split`, and those labels."""
    data_dir = Path(data_dir)
    corpus = load_corpus(data_dir / "corpus.jsonl")
    all_queries = load_queries(data_dir / "queries.jsonl")
    qrels = load_qrels(data_dir / "qrels" / f"{split}.tsv")

    missing = set(qrels) - set(all_queries)
    if missing:
        raise ValueError(f"{len(missing)} labelled queries are missing from queries.jsonl")

    queries = {qid: all_queries[qid] for qid in qrels}
    return corpus, queries, qrels