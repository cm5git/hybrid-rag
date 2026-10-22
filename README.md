# Hybrid RAG over SciFact

Retrieval-augmented question answering over 5,183 biomedical abstracts (the SciFact corpus from BEIR). It combines keyword search (BM25, written from scratch) and dense embeddings (bge-small-en-v1.5) with reciprocal rank fusion, then generates answers that cite the passages they use, and declines when the passages do not contain the evidence.

**Live demo:** https://hybrid-rag.bluewater-bf71659a.uksouth.azurecontainerapps.io

## What is measured, and what is not

### Retrieval (300 SciFact test claims)

| System | nDCG@10 |
|---|---|
| BM25 (from scratch) | 0.660 |
| Hybrid (BM25 + dense, RRF k=60) | 0.708 |

Hybrid recall@100 is 0.965, so the right paper is almost always somewhere in the candidate set. Differences were checked with a paired bootstrap rather than read off a single number.

Two things that did **not** work, kept because they are real results:
- A cross-encoder reranker did not help (change of -0.018 nDCG@10, within noise), so the demo does not use it.
- LLM query rewriting was implemented but never evaluated, so I make no claim about it.

### Answer quality (a smoke test, not a benchmark)

The generator sees only the top 5 retrieved passages, must cite them as [1], [2], and must reply `INSUFFICIENT_EVIDENCE` when they do not support an answer. Code then checks the citations: an answer with no valid citation, or one citing a passage that does not exist, is withheld.

Final run, 15 SciFact test claims not used for any tuning, plus 8 off-topic questions:

| | Answered | Declined |
|---|---|---|
| Off-topic questions (should decline) | 0 / 8 | 8 / 8 |
| In-corpus claims | 6 / 15 | 9 / 15 |

An LLM judge checked each answer against the passages it cited: 6 of 6 supported.

How far to trust this:
- These are small samples. 6 of 15 answered has a 95% range of roughly 20% to 64%, and 6 of 6 faithful is consistent with a true rate as low as about 61%. 0 of 8 off-topic answered does not show the system never answers off-topic questions.
- The judge and the generator are the same model family, so the judge may be lenient.
- On the training claims the answered rate was lower (2 of 15), and using 8 passages instead of 5 did not help (1 of 15).

Why so many declines: I read the eight declined train claims whose relevant paper was in the passages. In six or seven the abstract simply does not state the claim (SciFact labels papers by their full text, but the system only sees abstracts), so declining was correct. One was a real miss: the abstract supported the claim, but the claim said "2001" and the survey was run in 2000. I did not loosen the prompt to rescue it, since a looser rule would likely make the system answer things it should not.

A bug found along the way: truncating abstracts to 1,200 characters made 67% of them lose text and caused most early declines. Raising it to 3,000 fixed that.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt fastapi uvicorn httpx
python scripts/download_data.py              # SciFact into data/scifact
echo "GOOGLE_API_KEY=your-key" > .env        # optional: without it the app is a search engine
uvicorn app.server:app --port 8000
pytest -q                                    # 50 tests, no model downloads needed
```

Answer evaluation: `python scripts/eval_answers.py --split train --n-passages 5` (tune on train; use `--split test --offset 15` once at the end).

Docker: `docker build -t hybrid-rag . && docker run -p 7860:7860 -e GOOGLE_API_KEY=... hybrid-rag`

## Demo limits

The demo uses Gemini's free tier, which allows about 15 requests per minute. The app limits each visitor to 6 requests a minute and caps model calls overall per minute. When a limit is reached it still returns the retrieved passages and says so.

## Layout

- `src/` retrieval (BM25, dense, hybrid), metrics, evaluation, cited generation, faithfulness judge
- `app/` FastAPI service and the web page
- `scripts/` evaluation and diagnostic scripts
- `tests/` unit tests with fake models