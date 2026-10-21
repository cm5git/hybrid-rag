from src.faithfulness import judge, parse_verdict, summarise
from tests.test_generate import FakeLLM, gen


def test_parse_verdict_handles_unsupported_before_supported():
    assert parse_verdict("UNSUPPORTED.") == "UNSUPPORTED"
    assert parse_verdict("Supported") == "SUPPORTED"
    assert parse_verdict("partial") == "PARTIAL"
    assert parse_verdict("no idea") == "UNPARSEABLE"


def test_summarise_counts_and_rate():
    s = summarise(["SUPPORTED", "SUPPORTED", "PARTIAL", "UNSUPPORTED"])
    assert s["n"] == 4 and s["supported_rate"] == 0.5
    assert summarise([])["supported_rate"] == 0.0


def test_judge_sees_only_cited_passages():
    a = gen(FakeLLM("Aspirin [1].")).ask("q")
    j = FakeLLM("SUPPORTED")
    assert judge(j, a) == "SUPPORTED"
    assert "Aspirin lowers" in j.prompts[0]
    assert "Unrelated" not in j.prompts[0]  # uncited passages are not shown to the judge