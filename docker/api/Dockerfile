FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

COPY alembic.ini ./
COPY migrations ./migrations
COPY config ./config
COPY docker/api/entrypoint.sh /entrypoint.sh

RUN useradd --system --uid 1000 rag && mkdir -p /data && chown rag /data
USER rag

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
