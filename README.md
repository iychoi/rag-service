# RAG Service

A REST service that manages documents in datasets and provides parsing (Docling, MinerU) → chunking → embedding → pgvector storage → search.
See [API.md](API.md) for the REST API reference, [openapi.yaml](openapi.yaml) for the OpenAPI (Swagger) spec, and [TODO.md](TODO.md) for remaining work.

## Running

```bash
docker compose up -d --build      # rag-api(:8000), postgres+pgvector(:5432), docling-serve CPU(:5001), mineru-api CPU(:8010)
curl localhost:8000/readyz
curl localhost:8000/v1/parsers
```

API docs: http://localhost:8000/docs (Swagger UI), spec at http://localhost:8000/openapi.json

Parser selection: set `parser` to `docling` or `mineru` in the indexing request.
MinerU runs with the `basic` tier on CPU; per request you can pass `options={"tier": "flash"|"basic", "ocr_mode": "auto"|"txt"|"ocr", "page_range": "1-5"}`.
To move parsers to GPU, see [Parsers on GPU](#parsers-on-gpu).

## Parsers on GPU

Parsers run as separate containers and are reached over HTTP, so moving them needs configuration only, no code changes.

| Scenario | How |
|----------|-----|
| GPU on the same host | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build` — rebuilds MinerU with torch/vLLM (`MINERU_EXTRAS=full`, `MINERU_MODEL_TIER=standard`), sets `MINERU_TIER=standard`, and switches Docling to the CUDA image. **Untested template.** |
| Parsers on a separate GPU node | Run the parser containers there and point rag-api's `MINERU_URL` / `DOCLING_URL` (and `MINERU_TIER`) at them. |
| MinerU on CPU, VLM on a GPU node | Set `MINERU_VLM_SERVER_URL` on the mineru container and `MINERU_TIER=standard`. |

MinerU tiers: `flash` (text layer only), `basic` (ONNX models, CPU default), `standard` / `advanced` (VLM, GPU or remote VLM). The tier used is recorded in each job's `options` and the document's `parser_backend`, so changing the default tier also invalidates the parse cache.

## Usage

```bash
B=http://localhost:8000/v1

# Create a dataset (embedding model is required)
curl -s $B/embedding-models
DS=$(curl -s -X POST $B/datasets -H 'content-type: application/json' \
  -d '{"name":"legal","embedding_model_id":"hash-dev"}' | jq -r .id)

# Upload + index
JOB=$(curl -s -X POST $B/datasets/$DS/documents \
  -F file=@contract.pdf -F index=true -F parser=docling -F 'options={"do_ocr": false}' \
  -F 'metadata={"year": 2025}' | jq -r .job.id)
curl -s $B/datasets/$DS/jobs/$JOB          # status / stage / progress

# Search
curl -s -X POST $B/datasets/$DS/search -H 'content-type: application/json' \
  -d '{"query":"contract termination terms","top_k":5,"filters":{"metadata":{"year":2025}}}'
```

If the embedding provider requires a key (after the LiteLLM integration), add the `X-Model-API-Key` header to indexing and search requests.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest                          # requires Docker (testcontainers starts pgvector)
DOCLING_URL=http://localhost:5001 MINERU_URL=http://localhost:8010 .venv/bin/pytest -m integration   # against real parsers
.venv/bin/python scripts/export_openapi.py   # regenerate openapi.yaml after API changes (a test checks it is current)
```

To run only the API locally:

```bash
docker compose up -d postgres docling mineru
.venv/bin/alembic upgrade head
.venv/bin/uvicorn --factory rag_service.main:create_app --reload
```
