# TODO

Remaining work, grouped by phase. Each item carries the context needed to pick it up.

Current state: dataset/document management, indexing jobs (Docling and MinerU on CPU), per-dataset pgvector tables, search, and `/query` (LLM answers with citations via LiteLLM) all work end to end. Each dataset picks its embedding model at creation (a LiteLLM model name, or the development-only `hash` provider); the LiteLLM base URL is server configuration and callers send only their key. The LiteLLM paths are tested against a mocked LiteLLM only.

---

## Phase 1 — MVP (remaining)

### LiteLLM embedding provider
- [x] `litellm` provider (`OpenAILikeEmbedding`, base URL from `LITELLM_BASE_URL`, key per request); datasets choose the model at creation and the output dimension is measured with one embedding call. The server-side model catalog was removed.
- [x] Upstream errors mapped without leaking the key (`providers/upstream.py`): `502 upstream_auth_failed` / `upstream_error` / `upstream_unreachable`, `504 upstream_timeout`; indexing jobs fail with `embedding failed (<code>): ...`.
- [ ] **Verify against the real LiteLLM** once an embedding model is registered: dataset creation (dimension probe), indexing, search. Candidate models (all ≤ 2000 dims, the pgvector HNSW limit):

  | LiteLLM name (suggested) | Bedrock model id | Dim | Max input | Asymmetric (`input_type`) |
  |--------------------------|------------------|-----|-----------|---------------------------|
  | `titan-embed-v2` | `amazon.titan-embed-text-v2:0` | 1024 (256/512 via `dimensions`) | 8,192 tokens | no — default |
  | `cohere-embed-v4` | `cohere.embed-v4:0` | 1536 | 128K tokens | yes: `{"document": "search_document", "query": "search_query"}` |
  | `cohere-embed-multilingual-v3` | `cohere.embed-multilingual-v3` | 1024 | 512 tokens | yes — fallback if v4 is unavailable |

- [ ] **Chunk size vs. model input limit**: the catalog used to reject a `chunk_size` above the model's input limit; now an oversized chunk only fails the indexing job. Consider reading `max_input_tokens` from LiteLLM's `/model/info` at dataset creation.
- [ ] **Model discovery**: `GET /v1/embedding-models` was removed with the catalog. If clients need a model list, proxy LiteLLM's `/v1/models` with the caller's key.
- [ ] **Dev-only `hash` provider**: consider a setting to disable it in production.

### Blocked / to verify
- [ ] **Blocker:** ask the LiteLLM admin to register an embedding model (at least Titan Text Embeddings V2; Cohere Embed v4 too, for comparison). As of 2026-11-04 LiteLLM only has chat models registered (all on Bedrock, `us.` cross-region profiles).
- [ ] Verify that LiteLLM forwards Cohere's `input_type` (sent as `extra_body`) to Bedrock.
- [ ] Verify Cohere Embed v4 availability in the US region in use.

---

## Phase 2 — Quality

- [ ] **Cohere Embed v4 vs. Titan V2**: `input_type` per dataset is implemented (sent as `extra_body`); compare Korean retrieval quality and decide the default model.
- [x] **`POST /v1/datasets/{dataset_id}/query`**: retrieval + LLM answer with `[n]` citations (`services/answering.py`, `providers/llms.py`). Mocked-LiteLLM tests only.
  - Settings: `LITELLM_BASE_URL`, `LLM_MODEL=claude-sonnet-5-5`, `LLM_ALLOWED_MODELS`, `LLM_TIMEOUT_S`, `LLM_MAX_OUTPUT_TOKENS`.
  - Models currently in LiteLLM and their price per 1M tokens (input/output): `claude-sonnet-5-5` $2.20/$11.00, `claude-haiku-5-5` $0.11/$0.55, `claude-opus-5-5` $4.40/$22.00 (too costly as default).
- [ ] **Verify `/query` against the real LiteLLM** (needs the LiteLLM base URL and a key): check the answer language, citation format, and token usage reporting with `claude-sonnet-5-5`.
- [ ] **`/query` follow-ups**: streaming responses (SSE), and a prompt-size budget (today all `top_k` chunks are sent; with large chunks and `top_k` up to 50 this can get expensive).
- [ ] **Query rewriting / contextual retrieval** with `claude-haiku-5-5`, e.g. add a short document-context summary to each chunk at indexing time (needs an `LLM_FAST_MODEL` setting).
- [ ] **Structure-aware chunking**:
  - Docling: the DoclingDocument JSON is already cached as `raw.json` in the parse cache; use `DoclingNodeParser` (`llama-index-node-parser-docling`, needs only `docling-core`, no torch).
  - MinerU: `structured_content.json` (cached as `raw.json`) has per-page typed blocks with bboxes; use it to keep tables as single chunks and get exact page numbers.
- [ ] **Hybrid search + reranking**: `PGVectorStore(hybrid_search=True)` uses PostgreSQL full-text search, whose default configs do not tokenize Korean; evaluate `pg_bigm` / `pgroonga` or a separate BM25 index. Evaluate a reranker (Bedrock rerank models via LiteLLM).

---

## Phase 3 — Operations

- [ ] **Orphan vector cleanup**: `PGVectorStore` writes in its own session, so metadata and vector writes are not one transaction. Search already hides vectors whose document row is gone; add a periodic cleanup. If this becomes a problem, replace `PGVectorStore` with a vector store that shares the metadata session.
- [ ] **Structured logging** with header masking (`X-Model-API-Key` must never be logged) and **metrics** (parse duration per parser/tier, embedding tokens, job counts by status).
- [ ] **GPU validation**: `docker-compose.gpu.yml` is an untested template (MinerU rebuilt with `MINERU_EXTRAS=full`, `MINERU_MODEL_TIER=standard`, `MINERU_TIER=standard`; Docling on `docling-serve-cu128:v1.38.2`). Validate on a GPU host, and validate running parsers on a remote node via `MINERU_URL` / `DOCLING_URL`.
- [ ] **MinerU tier check at request time**: an unsupported `tier` currently fails the job; check against the server's `GET /v1/tiers` when the job is created and return `422`.
- [ ] **Stream uploads to MinerU**: the MinerU client reads the whole file into memory before `PUT`; stream it instead (uploads can be up to `MAX_UPLOAD_MB`).
- [ ] **Parser version bumps**: docling-serve (`v1.38.2`) and MinerU (`4.1.7`) are pinned because their HTTP APIs change between versions (MinerU 2.x `/file_parse` → 4.x `/v1/tasks`). When upgrading, run `pytest -m integration` against the new containers and re-record the fixtures in `tests/fixtures/`.

---

## Phase 4 — Scale and extensions

- [ ] **External worker queue** (arq or Celery + Redis) so workers scale independently of the API. API keys must then cross process boundaries: pass them encrypted with a short TTL, never persist them.
- [ ] **Progress notifications** via SSE or webhooks (today clients poll the job).
- [ ] **S3-compatible BlobStore** behind the existing `BlobStore` interface. Parsers could fetch inputs by presigned URL (MinerU tasks accept a source `url`; docling-serve has `/v1/convert/source`).
- [ ] **Multimodal**: images and tables (Cohere Embed v4 image embeddings, LLM captioning).
- [ ] **Multi-dataset search** (only if needed): per-dataset tables stay; fan out and merge. Same embedding model → embed the query once and merge by cosine score (a single `UNION ALL` query across tables works). Different models → merge with RRF or a reranker. API shape: `POST /v1/search` with explicit `dataset_ids` (or `?dataset_ids=a,b` if the gateway can only inspect the URL); no "search everything".
- [ ] **Dedicated vector DB** if pgvector stops scaling (thousands of datasets/tables or large tables); `VectorStoreManager` is the seam to swap.

---

## Open questions

1. Can the LiteLLM admin register Titan Text Embeddings V2 (and Cohere Embed v4)? Once registered, put the LiteLLM model names into `config/embedding_models.yaml`.
2. The shared Gemini conversation (https://share.gemini.google/I6QJtcFEpbM4) could not be read (JavaScript-rendered page). If it contains decisions (e.g. chunking strategy) that differ from the current implementation, they still need to be reconciled.
