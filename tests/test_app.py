import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import Limits, create_app  # noqa: E402
from src.generate import Generator  # noqa: E402
from tests.test_generate import CORPUS, FakeLLM, FakeRetriever  # noqa: E402


def client(llm, **limits):
    g = Generator(FakeRetriever(), CORPUS, llm, n_passages=3)
    return TestClient(create_app(g, Limits(**limits)))


def test_health():
    r = client(FakeLLM("x [1]")).get("/health")
    assert r.json() == {"status": "ok", "documents": 3, "llm_enabled": True}


def test_ask_answered():
    r = client(FakeLLM("Aspirin lowers risk [1].")).post(
        "/ask", json={"question": "does aspirin help?"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered" and body["cited"] == ["d1"]


def test_ask_rejects_short_question():
    assert client(None).post("/ask", json={"question": "a"}).status_code == 422


def test_llm_cap_falls_back_to_passages():
    c = client(FakeLLM("Aspirin [1]."), llm_per_minute=1, per_ip_per_minute=50)
    assert c.post("/ask", json={"question": "first one"}).json()["status"] == "answered"
    second = c.post("/ask", json={"question": "second one"}).json()
    assert second["status"] == "retrieval_only" and "limit" in second["note"]


def test_rate_limit_per_ip():
    c = client(None, per_ip_per_minute=2)
    codes = [c.post("/ask", json={"question": "some question"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]

def test_index_page_served():
    r = client(None).get("/")
    assert r.status_code == 200 and "Hybrid RAG" in r.text