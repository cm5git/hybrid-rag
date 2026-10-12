import pytest

from src.evaluate import bootstrap_ci, evaluate, format_report, summarise

QUERIES = {"q1": "alpha", "q2": "beta"}
QRELS = {"q1": {"d1"}, "q2": {"d2", "d3"}}
ALL_DOCS = ["d1", "d2", "d3", "d4", "d5"]


def perfect(query_text):
    relevant = QRELS[{"alpha": "q1", "beta": "q2"}[query_text]]
    return sorted(relevant) + [d for d in ALL_DOCS if d not in relevant]


def returns_nothing(query_text):
    return []


def test_perfect_retriever_scores_one():
    per_query = evaluate(perfect, QUERIES, QRELS)
    for values in per_query.values():
        assert values == pytest.approx([1.0, 1.0])


def test_empty_retriever_scores_zero():
    per_query = evaluate(returns_nothing, QUERIES, QRELS)
    for values in per_query.values():
        assert values == [0.0, 0.0]


def test_one_score_per_query():
    per_query = evaluate(perfect, QUERIES, QRELS)
    assert all(len(v) == len(QUERIES) for v in per_query.values())


def test_depth_truncates_results():
    # d1 is ranked 3rd; with depth=2 it is cut off and cannot be found.
    def late(query_text):
        return ["d4", "d5", "d1"]

    per_query = evaluate(late, {"q1": "alpha"}, {"q1": {"d1"}}, depth=2)
    assert per_query["recall@100"] == [0.0]


def test_bootstrap_of_constant_values_is_degenerate():
    low, high = bootstrap_ci([0.5] * 50)
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)


def test_bootstrap_interval_contains_mean_and_is_reproducible():
    values = [0.0, 1.0] * 50
    low, high = bootstrap_ci(values)
    assert low < 0.5 < high
    assert bootstrap_ci(values) == (low, high)  # same seed, same answer


def test_summarise_and_report():
    per_query = evaluate(perfect, QUERIES, QRELS)
    summary = summarise(per_query, n_boot=50)
    assert summary["ndcg@10"]["mean"] == pytest.approx(1.0)
    report = format_report(summary, "perfect", len(QUERIES))
    assert "ndcg@10" in report and "perfect" in report