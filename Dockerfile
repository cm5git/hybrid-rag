FROM python:3.10-slim

# Hugging Face Spaces runs containers as user 1000 and expects port 7860.
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user requirements-app.txt .
# CPU-only torch keeps the image small; the pins match requirements.txt.
RUN pip install --no-cache-dir --user torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements-app.txt

COPY --chown=user . .
# Fetch the dataset and precompute embeddings at build time.
RUN python scripts/download_data.py && python scripts/warm_cache.py

EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
