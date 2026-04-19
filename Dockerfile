FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    VECTOR_DB_PATH=/app/.local/vector_store

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY examples ./examples

RUN pip install .

RUN mkdir -p /app/reports /app/.local && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

ENTRYPOINT ["compare-providers"]
