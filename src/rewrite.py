"""LLM query rewriting before the query reaches the index (with a disk cache)."""
import json
import os
import time
from pathlib import Path

PROMPT = """You rewrite a scientific claim into a search query for a collection of \
biomedical research abstracts. Keep every specific entity, number and technical term. \
Expand abbreviations and add one or two likely synonyms. Do not answer the claim. \
Output only plain space-separated words and phrases on a single line. \
No AND/OR operators, no parentheses, no quotes.

Claim: {query}
Query:"""


class QueryRewriter:
    def __init__(self, cache_path: str = "data/cache/rewrites.json",
                 model: str = "gemini-3.5-flash-lite", pause: float = 6.5):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model
        self.pause = pause  # seconds between API calls, to respect free-tier limits
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}

    def rewrite(self, query: str) -> str:
        if query in self.cache:
            return self.cache[query]
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=PROMPT.format(query=query)
            )
            out = (resp.text or "").strip().splitlines()[0].strip()
        except Exception as exc:
            # Fail loudly: a silent fallback would make a broken run look valid.
            raise RuntimeError(f"rewrite failed for {query!r}: {exc!r}") from exc
        self.cache[query] = out or query
        self.path.write_text(json.dumps(self.cache, indent=1))
        time.sleep(self.pause)
        return self.cache[query]


class RewriteThenSearch:
    def __init__(self, rewriter: QueryRewriter, base):
        self.rewriter = rewriter
        self.base = base

    def search(self, query: str, k: int = 100) -> list[str]:
        return self.base.search(self.rewriter.rewrite(query), k)