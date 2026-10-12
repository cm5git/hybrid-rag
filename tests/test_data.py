import json

import pytest

from src.data import load_split


def write_dataset(tmp_path, qrels_rows, queries=None):
    (tmp_path / "qrels").mkdir()
    corpus = [
        {"_id": "10", "title": "T10", "text": "alpha"},
        {"_id": "20", "title": "T20", "text": "beta"},
        {"_id": "30", "title": "T30", "text": "gamma"},
    ]
    queries = queries or [
        {"_id": "1", "text": "first claim"},
        {"_id": "2", "text": "second claim"},
        {"_id": "3", "text": "claim only in train"},
    ]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in corpus))
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in queries))
    header = "query-id\tcorpus-id\tscore\n"
    body = "\n".join("\t".join(r) for r in qrels_rows)
    (tmp_path / "qrels" / "test.tsv").write_text(header + body + "\n")


def test_load_split_keeps_only_labelled_queries(tmp_path):
    write_dataset(tmp_path, [("1", "10", "1"), ("2", "20", "1"), ("2", "30", "1")])
    corpus, queries, qrels = load_split(tmp_path, "test")

    assert set(corpus) == {"10", "20", "30"}
    assert set(queries) == {"1", "2"}  # query 3 has no test labels
    assert qrels == {"1": {"10"}, "2": {"20", "30"}}


def test_ids_are_strings(tmp_path):
    write_dataset(tmp_path, [("1", "10", "1")])
    corpus, queries, qrels = load_split(tmp_path, "test")
    assert all(isinstance(k, str) for k in corpus)
    assert all(isinstance(k, str) for k in queries)


def test_zero_scores_are_ignored(tmp_path):
    write_dataset(tmp_path, [("1", "10", "1"), ("1", "20", "0")])
    _, _, qrels = load_split(tmp_path, "test")
    assert qrels == {"1": {"10"}}


def test_label_without_query_raises(tmp_path):
    write_dataset(tmp_path, [("99", "10", "1")])
    with pytest.raises(ValueError):
        load_split(tmp_path, "test")