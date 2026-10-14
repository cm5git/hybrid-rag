from src.hybrid import HybridRetriever, rrf


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b"], ["c", "b"]])
    # b is 2nd in both lists: 2/62 beats a or c, each 1st in only one list: 1/61
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}


def test_rrf_exact_value():
    fused = rrf([["a"], ["a"]], k=60)
    assert fused == ["a"]


def test_rrf_depth():
    assert len(rrf([["a", "b", "c"]], depth=2)) == 2


class Fake:
    def __init__(self, out):
        self.out = out

    def search(self, q, k=100):
        return self.out[:k]


def test_hybrid_retriever():
    h = HybridRetriever([Fake(["x", "y"]), Fake(["y", "z"])])
    assert h.search("q")[0] == "y"