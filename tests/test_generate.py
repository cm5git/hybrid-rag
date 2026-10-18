from src.generate import Answer, CachedLLM, Generator, LLMUnavailable, parse_answer


class FakeRetriever:
    def search(self, query, k=100):
        return ["d1", "d2", "d3"][:k]


CORPUS = {
    "d1": {"title": "Aspirin", "text": "Aspirin lowers colorectal cancer risk."},
    "d2": {"title": "Vitamin D", "text": "Vitamin D and fractures."},
    "d3": {"title": "Other", "text": "Unrelated text."},
}


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


def gen(llm, n=3):
    return Generator(FakeRetriever(), CORPUS, llm, n_passages=n)


def test_parse_valid_citation():
    assert parse_answer("Aspirin helps [1][3].", 3) == ("answered", "Aspirin helps [1][3].", [1, 3])


def test_parse_decline_wins():
    assert parse_answer("INSUFFICIENT_EVIDENCE", 3)[0] == "declined"


def test_parse_no_citation_is_withheld():
    assert parse_answer("Aspirin helps.", 3)[0] == "uncited"


def test_parse_out_of_range_citation_is_withheld():
    assert parse_answer("Aspirin helps [7].", 3)[0] == "uncited"
    assert parse_answer("Aspirin helps [1] and more [9].", 3)[0] == "uncited"


def test_answered_maps_numbers_to_doc_ids():
    a = gen(FakeLLM("Aspirin lowers risk [1]. Vitamin D too [2].")).ask("q")
    assert a.status == "answered" and a.cited == ["d1", "d2"]


def test_prompt_contains_only_retrieved_passages():
    llm = FakeLLM("x [1]")
    gen(llm, n=2).ask("a question")
    assert "[1] Aspirin." in llm.prompts[0] and "[2] Vitamin D." in llm.prompts[0]
    assert "Unrelated" not in llm.prompts[0]


def test_declined_has_no_text():
    a = gen(FakeLLM("INSUFFICIENT_EVIDENCE")).ask("q")
    assert a.status == "declined" and a.text == "" and a.cited == []


def test_llm_unavailable_falls_back_to_passages():
    a = gen(FakeLLM(error=LLMUnavailable("quota"))).ask("q")
    assert a.status == "retrieval_only" and len(a.passages) == 3


def test_no_llm_is_retrieval_only():
    assert gen(None).ask("q").status == "retrieval_only"
    assert gen(FakeLLM("x [1]")).ask("q", use_llm=False).status == "retrieval_only"


def test_cached_llm_calls_once(tmp_path):
    inner = FakeLLM("hello")
    c = CachedLLM(inner, str(tmp_path / "c.json"))
    assert c.complete("p") == "hello" and c.complete("p") == "hello"
    assert len(inner.prompts) == 1
    again = CachedLLM(FakeLLM("different"), str(tmp_path / "c.json"))
    assert again.complete("p") == "hello"  # survives a restart


def test_answer_to_dict():
    assert Answer("q", "declined").to_dict()["status"] == "declined"