"""Cited answer generation with a decline check.

The model sees only the top retrieved passages. It must cite them, and it must say
INSUFFICIENT_EVIDENCE when they do not contain the answer. An answer that cites nothing,
or cites a passage number that does not exist, is withheld rather than shown.
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.common import doc_text

DECLINE_TOKEN = "INSUFFICIENT_EVIDENCE"
CITE_RE = re.compile(r"\[(\d+)\]")

PROMPT = """You answer a question or check a claim using ONLY the numbered passages below, \
which are research abstracts.

Rules:
- Use only facts stated in the passages. Do not add outside knowledge.
- Every sentence must end with a citation such as [1] or [2][3] naming the passage(s) that support it.
- If the passages do not contain enough evidence, reply with exactly: {decline}
- Keep the answer under 120 words.

Passages:
{context}

Question or claim: {question}
Answer:"""


class LLMUnavailable(RuntimeError):
    """Raised when the language model cannot answer (quota, network, missing key)."""


class GeminiLLM:
    def __init__(self, model: str = "gemini-3.5-flash-lite"):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMUnavailable("GOOGLE_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.model = model

    def complete(self, prompt: str) -> str:
        try:
            resp = self.client.models.generate_content(model=self.model, contents=prompt)
            return (resp.text or "").strip()
        except Exception as exc:  # quota, network, safety block
            raise LLMUnavailable(repr(exc)) from exc


class CachedLLM:
    """Wrap any LLM with an on-disk cache, so an interrupted evaluation can resume."""

    def __init__(self, llm, path: str = "data/cache/llm_cache.json"):
        self.llm = llm
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.calls = 0  # real (uncached) calls made by this process

    def complete(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        out = self.llm.complete(prompt)
        self.calls += 1
        self.cache[key] = out
        self.path.write_text(json.dumps(self.cache))
        return out


@dataclass
class Answer:
    question: str
    status: str  # answered | declined | uncited | retrieval_only
    text: str = ""
    cited: list = field(default_factory=list)  # doc ids the answer cites
    passages: list = field(default_factory=list)  # [{"n", "id", "title", "text"}]
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def parse_answer(raw: str, n_passages: int):
    """Return (status, text, cited_numbers) from the raw model output."""
    text = raw.strip()
    if DECLINE_TOKEN in text:
        return "declined", "", []
    nums = [int(m) for m in CITE_RE.findall(text)]
    valid = sorted({n for n in nums if 1 <= n <= n_passages})
    invalid = [n for n in nums if not 1 <= n <= n_passages]
    if not valid or invalid:
        return "uncited", "", []
    return "answered", text, valid


class Generator:
    def __init__(self, retriever, corpus: dict, llm=None, n_passages: int = 5,
                 max_chars: int = 3000):
        self.retriever = retriever
        self.corpus = corpus
        self.llm = llm
        self.n_passages = n_passages
        self.max_chars = max_chars

    def passages_for(self, question: str) -> list:
        ids = self.retriever.search(question, self.n_passages)
        out = []
        for n, doc_id in enumerate(ids, start=1):
            doc = self.corpus[doc_id]
            out.append({
                "n": n, "id": doc_id,
                "title": doc.get("title", "") if isinstance(doc, dict) else "",
                "text": doc_text(doc)[: self.max_chars],
            })
        return out

    def ask(self, question: str, use_llm: bool = True) -> Answer:
        passages = self.passages_for(question)
        if not use_llm or self.llm is None:
            return Answer(question, "retrieval_only", passages=passages,
                          note="Showing the retrieved passages only; answer generation is off.")
        context = "\n\n".join(f"[{p['n']}] {p['text']}" for p in passages)
        prompt = PROMPT.format(decline=DECLINE_TOKEN, context=context, question=question)
        try:
            raw = self.llm.complete(prompt)
        except LLMUnavailable as exc:
            return Answer(question, "retrieval_only", passages=passages,
                          note=f"The language model is unavailable ({str(exc)[:80]}). "
                               "Showing the retrieved passages only.")
        status, text, nums = parse_answer(raw, len(passages))
        note = {
            "declined": "The retrieved passages do not contain enough evidence to answer.",
            "uncited": "The model's answer cited no valid passage, so it was withheld.",
        }.get(status, "")
        cited = [passages[n - 1]["id"] for n in nums]
        return Answer(question, status, text=text, cited=cited, passages=passages, note=note)