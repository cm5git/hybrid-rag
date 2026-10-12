import pytest

from src.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

RANKING = ["a", "b", "c", "d", "e"]


def test_single_relevant_at_rank_3():
    relevant = {"c"}
    assert recall_at_k(RANKING, relevant, 3) == pytest.approx(1.0)
    assert recall_at_k(RANKING, relevant, 2) == pytest.approx(0.0)
    assert reciprocal_rank(RANKING, relevant) == pytest.approx(1 / 3)
    assert ndcg_at_k(RANKING, relevant, 5) == pytest.approx(0.5)


def test_two_relevant_c_and_e():
    relevant = {"c", "e"}
    assert recall_at_k(RANKING, relevant, 3) == pytest.approx(0.5)
    assert reciprocal_rank(RANKING, relevant) == pytest.approx(1 / 3)
    assert ndcg_at_k(RANKING, relevant, 5) == pytest.approx(0.5437, abs=1e-3)


def test_two_relevant_b_and_d():
    relevant = {"b", "d"}
    assert precision_at_k(RANKING, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(RANKING, relevant, 3) == pytest.approx(0.5)
    assert recall_at_k(RANKING, relevant, 5) == pytest.approx(1.0)
    assert reciprocal_rank(RANKING, relevant) == pytest.approx(0.5)
    assert ndcg_at_k(RANKING, relevant, 5) == pytest.approx(0.6509, abs=1e-3)


def test_relevant_at_last_rank():
    relevant = {"e"}
    assert reciprocal_rank(RANKING, relevant) == pytest.approx(0.2)
    assert ndcg_at_k(RANKING, relevant, 5) == pytest.approx(0.3869, abs=1e-3)


def test_perfect_ranking_scores_one():
    relevant = {"a"}
    assert recall_at_k(RANKING, relevant, 1) == pytest.approx(1.0)
    assert reciprocal_rank(RANKING, relevant) == pytest.approx(1.0)
    assert ndcg_at_k(RANKING, relevant, 5) == pytest.approx(1.0)


def test_relevant_document_not_returned():
    relevant = {"z"}
    assert recall_at_k(RANKING, relevant, 5) == 0.0
    assert reciprocal_rank(RANKING, relevant) == 0.0
    assert ndcg_at_k(RANKING, relevant, 5) == 0.0


def test_empty_relevant_set_returns_zero():
    assert recall_at_k(RANKING, set(), 5) == 0.0
    assert reciprocal_rank(RANKING, set()) == 0.0
    assert ndcg_at_k(RANKING, set(), 5) == 0.0


def test_empty_ranking_returns_zero():
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0
    assert ndcg_at_k([], {"a"}, 5) == 0.0


def test_ndcg_is_one_when_all_relevant_are_on_top():
    relevant = {"a", "b"}
    assert ndcg_at_k(RANKING, relevant, 5) == pytest.approx(1.0)


def test_ndcg_ideal_is_capped_at_k():
    # 3 relevant documents but k=2: the best possible top 2 is both relevant.
    relevant = {"a", "b", "c"}
    assert ndcg_at_k(RANKING, relevant, 2) == pytest.approx(1.0)