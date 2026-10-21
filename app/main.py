"""Web API. `create_app` takes a Generator, so tests can pass fakes instead of real models."""
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Limits:
    """Protects the free Gemini quota (15 requests per MINUTE, not per day).

    - per-IP limit: stops one visitor from using everything
    - global per-minute cap on model calls: when it is hit, we still return the retrieved
      passages (search results) instead of failing
    """

    def __init__(self, per_ip_per_minute: int = 6, llm_per_minute: int = 12):
        self.per_ip = per_ip_per_minute
        self.llm_cap = llm_per_minute
        self.ip_hits = defaultdict(deque)
        self.llm_hits = deque()
        self.lock = threading.Lock()

    @staticmethod
    def _trim(q: deque, now: float) -> None:
        while q and now - q[0] > 60:
            q.popleft()

    def allow_request(self, ip: str) -> bool:
        now = time.time()
        with self.lock:
            q = self.ip_hits[ip]
            self._trim(q, now)
            if len(q) >= self.per_ip:
                return False
            q.append(now)
            return True

    def take_llm_call(self) -> bool:
        now = time.time()
        with self.lock:
            self._trim(self.llm_hits, now)
            if len(self.llm_hits) >= self.llm_cap:
                return False
            self.llm_hits.append(now)
            return True


def create_app(generator, limits: Limits | None = None) -> FastAPI:
    app = FastAPI(title="Hybrid RAG over SciFact")
    limits = limits or Limits()

    @app.get("/health")
    def health():
        return {"status": "ok", "documents": len(generator.corpus),
                "llm_enabled": generator.llm is not None}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
        if not limits.allow_request(ip):
            raise HTTPException(status_code=429, detail="Too many requests; wait a minute.")
        use_llm = generator.llm is not None and limits.take_llm_call()
        answer = generator.ask(body.question, use_llm=use_llm)
        if generator.llm is not None and not use_llm:
            answer.note = ("The demo is at its per-minute limit for the free model. "
                           "Showing the retrieved passages only; try again in a minute.")
        return answer.to_dict()

    return app