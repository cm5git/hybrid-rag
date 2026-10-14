from src.bm25 import BM25, tokenize

CORPUS = {
    "d1": "Aspirin reduces the risk of heart attack.",
    "d2": "The heart pumps blood through the body.",
    "d3": "Quantum computers use qubits.",
    "d4": {"title": "Statins", "text": "Statins lower cholesterol and heart disease risk."},
}


def test_tokenize():
    assert tokenize("Hello, World-2!") == ["hello", "world", "2"]


def test_rare_word_beats_common_word():
    bm = BM25().fit(CORPUS)
    # 'aspirin' appears in one doc, 'heart' in three: the rare word should decide
    assert bm.search("aspirin heart", 4)[0] == "d1"


def test_no_overlap_scores_zero():
    bm = BM25().fit(CORPUS)
    assert bm.scores("zebra").sum() == 0


def test_title_is_searchable_and_k_limits_results():
    bm = BM25().fit(CORPUS)
    assert bm.search("statins", 1) == ["d4"]
    assert len(bm.search("heart", 2)) == 2


def test_repeating_a_word_has_diminishing_returns():
    corpus = {"a": "cat " * 3 + "dog", "b": "cat " * 30 + "dog", "c": "bird"}
    bm = BM25().fit(corpus)
    s = bm.scores("cat")
    assert s[1] < 3 * s[0]  # saturation: 10x the repeats is far from 10x the score