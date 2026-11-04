# RAG Service REST API

Version: v1 (rag-service 0.1.0)

This document specifies the REST API that rag-service provides: dataset and document management, indexing jobs, and search.
The OpenAPI (Swagger) spec is in [openapi.yaml](openapi.yaml); the running service also serves it at `/openapi.json`, with Swagger UI at `/docs`.

- [Conventions](#conventions)
- [Objects](#objects)
- [Endpoints](#endpoints)
  - [Embedding models](#embedding-models)
  - [Datasets](#datasets)
  - [Documents](#documents)
  - [Indexing and jobs](#indexing-and-jobs)
  - [Search](#search)
  - [Parsers](#parsers)
  - [Operations](#operations)
- [Parser options](#parser-options)
- [Error codes](#error-codes)
- [End-to-end example](#end-to-end-example)

---

## Conventions

### Base URL

All API endpoints are under `/v1`. The health endpoints (`/healthz`, `/readyz`) are not versioned.

```
http://localhost:8000/v1
```

The examples below use:

```bash
B=http://localhost:8000/v1
```

### Authentication and authorization

rag-service does **not** authenticate callers. An external gateway is expected to do that and to authorize access per **dataset**:

- Every endpoint that touches dataset contents has the dataset id in its path: `/v1/datasets/{dataset_id}/...`. A gateway can authorize by path prefix alone.
- A document or job id is only reachable under the dataset it belongs to. Using it under another dataset's path returns `404`.

### Model API key

Embedding (and later LLM) calls go through LiteLLM with a key supplied by the caller on each request. The service never stores the key.

| Header | Used by |
|--------|---------|
| `X-Model-API-Key` | Indexing requests (`POST .../index`, upload with `index=true`) and search |

The header is required only if the dataset's embedding model needs a key (`requires_api_key: true` in [`GET /embedding-models`](#list-embedding-models)). If it is required and missing, the request fails with `400 missing_model_api_key`.

For indexing, the key is kept in memory only while the job runs. If the server restarts mid-job, the job ends as `failed` with error `interrupted ...` and must be resubmitted with the key.

### IDs

Ids are 32-character lowercase hex strings (UUID4 without hyphens), for example `52967db904d44a09b33f88b3c9f64a6a`. A malformed id returns `404`.

### Timestamps

All timestamps are ISO 8601 in UTC, for example `2026-11-04T19:31:49.174949Z`.

### Pagination

List endpoints accept `limit` and `offset` query parameters and return a page object:

```json
{
  "items": [ ... ],
  "total": 2,
  "limit": 50,
  "offset": 0
}
```

Results are ordered newest first (`created_at` descending).

### Errors

Every error response has the same shape:

```json
{
  "error": {
    "code": "document_not_found",
    "message": "document '00000000000000000000000000000000' not found"
  }
}
```

`code` is stable and machine-readable. `message` is for humans and may change. See [Error codes](#error-codes) for the full list.

### Metadata

Datasets and documents accept a user-defined `metadata` object.

- Keys must match `^[A-Za-z][A-Za-z0-9_.-]{0,63}$`.
- Values must be a string, number, boolean, or `null`.
- These keys are reserved and rejected: `document_id`, `doc_id`, `ref_doc_id`, `filename`, `parser`, `page`, `section_path`, `chunk_index`, `job_id`, `header_path`, `_node_content`, `_node_type`.

Document metadata is copied onto every chunk, so it can be used as a [search filter](#search-a-dataset).

---

## Objects

### EmbeddingModel

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Catalog id. Pass it as `embedding_model_id` when creating a dataset. |
| `provider` | string | Implementation, e.g. `hash` (development) or `litellm`. |
| `model` | string | Model name at the provider. |
| `dim` | integer | Embedding dimension. |
| `max_input_tokens` | integer | Maximum input size. A dataset's `chunk_size` cannot exceed it. |
| `requires_api_key` | boolean | Whether requests must send `X-Model-API-Key`. |
| `description` | string \| null | |

### Dataset

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Dataset id. This is the unit of access control. |
| `name` | string | Display name. |
| `external_ref` | string \| null | Optional identifier from an external system. Unique. |
| `description` | string \| null | |
| `embedding_model_id` | string | Embedding model, fixed at creation. |
| `embedding_dim` | integer | Embedding dimension. |
| `chunk_size` | integer | Chunk size in tokens. |
| `chunk_overlap` | integer | Chunk overlap in tokens. |
| `metadata` | object | User metadata. |
| `created_at`, `updated_at` | string | |
| `document_counts` | object \| null | Document count per status plus `total`, e.g. `{"indexed": 1, "uploaded": 1, "total": 2}`. Included by get, create, and update; `null` in list responses. |

### Document

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Document id. |
| `dataset_id` | string | |
| `filename` | string | Original file name. |
| `mime_type` | string | |
| `size_bytes` | integer | |
| `sha256` | string | Content hash. Unique within a dataset. |
| `status` | string | `uploaded`, `indexing`, `indexed`, or `failed`. See [Document status](#document-status). |
| `parser` | string \| null | Parser that produced the current index (`docling`, `mineru`, or `passthrough`). |
| `parser_backend` | string \| null | Parser backend detail, e.g. `tier:basic` for MinerU. |
| `num_chunks` | integer \| null | Number of chunks in the current index. |
| `indexed_at` | string \| null | When the current index was built. |
| `last_job_id` | string \| null | Most recent indexing job. |
| `metadata` | object | User metadata. |
| `created_at`, `updated_at` | string | |
| `last_job` | Job \| null | The most recent job. Included by get and upload; `null` in list responses. |

#### Document status

| Status | Meaning |
|--------|---------|
| `uploaded` | Stored, never indexed (or the first job was canceled while queued). |
| `indexing` | A job is running. If the document was indexed before, the previous index stays searchable until the new one replaces it. |
| `indexed` | Searchable. |
| `failed` | The last job failed and there is no previous index. |

If re-indexing fails or is canceled, a document that was indexed before stays `indexed` with its previous chunks.

### Job

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Job id. |
| `dataset_id`, `document_id` | string | |
| `parser` | string | Parser actually used (`docling`, `mineru`, `passthrough`). |
| `options` | object | Effective parser options. For MinerU this always includes `tier`. |
| `force_reparse` | boolean | Whether the parse cache was bypassed. |
| `status` | string | `queued`, `running`, `succeeded`, `failed`, or `canceled`. |
| `stage` | string \| null | `parsing`, `chunking`, `embedding`, or `storing`. Last stage reached. |
| `progress` | number | 0.0 to 1.0. Parsing covers 0.0–0.4, chunking up to 0.45, embedding up to 0.9, storing up to 1.0. |
| `error` | string \| null | Failure reason when `status` is `failed` or `canceled`. |
| `created_at`, `started_at`, `finished_at` | string \| null | |

Job lifecycle:

```
queued ──► running ──► succeeded
   │          ├──────► failed
   └──────────┴──────► canceled
```

A document can have at most one active (`queued` or `running`) job.

### SearchResult

| Field | Type | Description |
|-------|------|-------------|
| `score` | number | Cosine similarity (higher is closer). |
| `text` | string | Chunk text (markdown). |
| `document_id` | string | |
| `filename` | string \| null | Current file name of the document. |
| `page` | integer \| null | 1-based page number, when the parser reports pages. |
| `section_path` | string[] | Heading path of the chunk, e.g. `["Terms of Service", "Article 5. Termination"]`. |
| `chunk_index` | integer \| null | Position of the chunk within the document. |
| `metadata` | object | Current document metadata. |

### Parser

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | `passthrough`, `docling`, or `mineru`. |
| `available` | boolean | Whether the parser service responds. |
| `detail` | string \| null | Extra status, e.g. the MinerU tiers the server offers, or a connection error. |
| `version` | string \| null | Parser service version. |
| `supported_extensions` | string[] | File extensions the parser accepts. |

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/embedding-models` | List embedding models |
| POST | `/v1/datasets` | Create a dataset |
| GET | `/v1/datasets` | List datasets |
| GET | `/v1/datasets/{dataset_id}` | Get a dataset |
| PATCH | `/v1/datasets/{dataset_id}` | Update a dataset |
| DELETE | `/v1/datasets/{dataset_id}` | Delete a dataset |
| POST | `/v1/datasets/{dataset_id}/documents` | Upload a document (optionally index it) |
| GET | `/v1/datasets/{dataset_id}/documents` | List documents |
| GET | `/v1/datasets/{dataset_id}/documents/{document_id}` | Get a document |
| PATCH | `/v1/datasets/{dataset_id}/documents/{document_id}` | Update document metadata |
| DELETE | `/v1/datasets/{dataset_id}/documents/{document_id}` | Delete a document |
| GET | `/v1/datasets/{dataset_id}/documents/{document_id}/content` | Download the original file |
| GET | `/v1/datasets/{dataset_id}/documents/{document_id}/parsed` | Get the parsed markdown |
| POST | `/v1/datasets/{dataset_id}/documents/{document_id}/index` | Index a document |
| POST | `/v1/datasets/{dataset_id}/index` | Index documents in bulk |
| GET | `/v1/datasets/{dataset_id}/jobs` | List jobs |
| GET | `/v1/datasets/{dataset_id}/jobs/{job_id}` | Get a job |
| POST | `/v1/datasets/{dataset_id}/jobs/{job_id}/cancel` | Cancel a job |
| POST | `/v1/datasets/{dataset_id}/search` | Search a dataset |
| GET | `/v1/parsers` | List parsers and their status |
| GET | `/healthz` | Liveness |
| GET | `/readyz` | Readiness |

---

### Embedding models

#### List embedding models

`GET /v1/embedding-models`

Returns the models a dataset can be created with.

```bash
curl -s $B/embedding-models
```

`200 OK`

```json
[
  {
    "id": "hash-dev",
    "provider": "hash",
    "model": "hash-ngram-v1",
    "dim": 512,
    "max_input_tokens": 8192,
    "requires_api_key": false,
    "description": "Local n-gram hashing embedding for development and tests"
  }
]
```

> `hash-dev` is a local embedding for development and tests. It only reflects lexical overlap. Bedrock models via LiteLLM (e.g. Titan Text Embeddings V2) will be added to the catalog.

---

### Datasets

#### Create a dataset

`POST /v1/datasets`

The embedding model is chosen here and cannot be changed later. The dataset's vector table is created immediately.

Request body:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | yes | | 1–255 characters. |
| `embedding_model_id` | string | yes | | An id from `GET /embedding-models`. |
| `external_ref` | string | no | `null` | Unique identifier from an external system. |
| `description` | string | no | `null` | |
| `chunk_size` | integer | no | `512` | 64–8192 tokens, and at most the model's `max_input_tokens`. |
| `chunk_overlap` | integer | no | `64` | 0–2048 tokens, and less than `chunk_size`. |
| `metadata` | object | no | `{}` | See [Metadata](#metadata). |

```bash
curl -s -X POST $B/datasets -H 'content-type: application/json' -d '{
  "name": "legal-docs",
  "embedding_model_id": "hash-dev",
  "external_ref": "team-legal-001",
  "description": "Contracts and policies",
  "chunk_size": 512,
  "chunk_overlap": 64,
  "metadata": {"team": "legal"}
}'
```

`201 Created`

```json
{
  "id": "52967db904d44a09b33f88b3c9f64a6a",
  "name": "legal-docs",
  "external_ref": "team-legal-001",
  "description": "Contracts and policies",
  "embedding_model_id": "hash-dev",
  "embedding_dim": 512,
  "chunk_size": 512,
  "chunk_overlap": 64,
  "metadata": {"team": "legal"},
  "created_at": "2026-11-04T19:31:49.174949Z",
  "updated_at": "2026-11-04T19:31:49.174960Z",
  "document_counts": {"total": 0}
}
```

Errors: `422 unknown_embedding_model`, `422 chunk_size_too_large`, `422 invalid_chunk_overlap`, `422 invalid_metadata`, `422 validation_error`, `409 external_ref_conflict`.

```json
{"error": {"code": "unknown_embedding_model", "message": "unknown embedding_model_id 'nope' (available: ['hash-dev'])"}}
```

#### List datasets

`GET /v1/datasets`

| Query | Type | Default | Description |
|-------|------|---------|-------------|
| `ids` | string | | Filter by id. Repeat the parameter or comma-separate: `?ids=a,b`. |
| `external_ref` | string | | Filter by external reference. |
| `limit` | integer | `50` | 1–200. |
| `offset` | integer | `0` | |

A gateway can pass the ids a caller may see in `ids` so that only those datasets are listed.

```bash
curl -s "$B/datasets?external_ref=team-legal-001"
curl -s "$B/datasets?ids=52967db904d44a09b33f88b3c9f64a6a,0123456789abcdef0123456789abcdef"
```

`200 OK` — a page of [Dataset](#dataset) objects (`document_counts` is `null`).

#### Get a dataset

`GET /v1/datasets/{dataset_id}`

```bash
curl -s $B/datasets/52967db904d44a09b33f88b3c9f64a6a
```

`200 OK`

```json
{
  "id": "52967db904d44a09b33f88b3c9f64a6a",
  "name": "legal-docs",
  "external_ref": "team-legal-001",
  "description": "Contracts and policies",
  "embedding_model_id": "hash-dev",
  "embedding_dim": 512,
  "chunk_size": 512,
  "chunk_overlap": 64,
  "metadata": {"team": "legal"},
  "created_at": "2026-11-04T19:31:49.174949Z",
  "updated_at": "2026-11-04T19:31:49.174960Z",
  "document_counts": {"indexed": 1, "uploaded": 1, "total": 2}
}
```

Errors: `404 dataset_not_found`.

#### Update a dataset

`PATCH /v1/datasets/{dataset_id}`

Only `name`, `description`, and `metadata` can change. The embedding model and chunking settings are fixed; other fields are rejected with `422 validation_error`. `metadata` replaces the whole object.

```bash
curl -s -X PATCH $B/datasets/52967db904d44a09b33f88b3c9f64a6a \
  -H 'content-type: application/json' \
  -d '{"description": "Contracts, policies, and amendments"}'
```

`200 OK` — the updated [Dataset](#dataset).

#### Delete a dataset

`DELETE /v1/datasets/{dataset_id}`

Cancels active jobs, then deletes the vector table, all documents, jobs, stored files, and parse caches. This cannot be undone.

```bash
curl -s -X DELETE $B/datasets/52967db904d44a09b33f88b3c9f64a6a
```

`204 No Content`

---

### Documents

#### Upload a document

`POST /v1/datasets/{dataset_id}/documents`

`multipart/form-data` fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | yes | | The document. Maximum size is `MAX_UPLOAD_MB` (default 100 MB). |
| `metadata` | string (JSON object) | no | `{}` | Document metadata. |
| `index` | boolean | no | `false` | Start an indexing job right after the upload. |
| `parser` | string | no | `docling` | `docling` or `mineru`. Used when `index=true`. |
| `options` | string (JSON object) | no | `{}` | [Parser options](#parser-options). Used when `index=true`. |

Headers: `X-Model-API-Key` when `index=true` and the dataset's model requires a key.

When `index=true`, the indexing request is validated (parser, file type, options, key, parser availability) **before** the file is stored, so a bad request leaves nothing behind.

`.md`, `.markdown`, and `.txt` files are always handled by the `passthrough` parser, whatever `parser` says; `options` are ignored for them.

Upload and index in one call:

```bash
curl -s -X POST $B/datasets/52967db904d44a09b33f88b3c9f64a6a/documents \
  -F file=@contract.pdf \
  -F 'metadata={"year": 2025, "category": "contract"}' \
  -F index=true \
  -F parser=docling \
  -F 'options={"do_ocr": false}'
```

`201 Created`

```json
{
  "document": {
    "id": "bbd713912b614848a8f0def3f73aa304",
    "dataset_id": "52967db904d44a09b33f88b3c9f64a6a",
    "filename": "contract.pdf",
    "mime_type": "application/pdf",
    "size_bytes": 3033,
    "sha256": "3c99bdb29e8d2ba87e7e7ff4d88664786e6128498e2e17327ac4ea727bb0c70c",
    "status": "uploaded",
    "parser": null,
    "parser_backend": null,
    "num_chunks": null,
    "indexed_at": null,
    "last_job_id": "2d18380d5e6548ed868383387a606e3b",
    "metadata": {"year": 2025, "category": "contract"},
    "created_at": "2026-11-04T19:31:49.280215Z",
    "updated_at": "2026-11-04T19:31:49.280232Z",
    "last_job": { "id": "2d18380d5e6548ed868383387a606e3b", "status": "queued", "...": "..." }
  },
  "job": {
    "id": "2d18380d5e6548ed868383387a606e3b",
    "dataset_id": "52967db904d44a09b33f88b3c9f64a6a",
    "document_id": "bbd713912b614848a8f0def3f73aa304",
    "parser": "docling",
    "options": {"do_ocr": false},
    "force_reparse": false,
    "status": "queued",
    "stage": null,
    "progress": 0.0,
    "error": null,
    "created_at": "2026-11-04T19:31:49.285116Z",
    "started_at": null,
    "finished_at": null
  }
}
```

Upload only (index later):

```bash
curl -s -X POST $B/datasets/52967db904d44a09b33f88b3c9f64a6a/documents -F file=@notes.txt
```

`201 Created` — `job` is `null` and `document.status` is `uploaded`.

Errors:

| Status | Code | When |
|--------|------|------|
| 409 | `duplicate_document` | A file with the same content already exists in the dataset. The message names the existing document id. |
| 413 | `file_too_large` | The file exceeds `MAX_UPLOAD_MB`. |
| 422 | `invalid_filename`, `invalid_json`, `invalid_metadata` | Bad form fields. |
| 422 | `unsupported_parser_or_format`, `invalid_parser_options` | With `index=true`: unknown parser, unsupported file type, or bad options. |
| 400 | `missing_model_api_key` | With `index=true`: the key is required but missing. |
| 503 | `parser_unavailable` | With `index=true`: the selected parser service is down. |

```json
{"error": {"code": "duplicate_document", "message": "identical file already exists as document 'bbd713912b614848a8f0def3f73aa304'"}}
```

#### List documents

`GET /v1/datasets/{dataset_id}/documents`

| Query | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | string | | `uploaded`, `indexing`, `indexed`, or `failed`. |
| `parser` | string | | `docling`, `mineru`, or `passthrough`. |
| `limit` | integer | `50` | 1–500. |
| `offset` | integer | `0` | |

```bash
curl -s "$B/datasets/52967db904d44a09b33f88b3c9f64a6a/documents?status=indexed&limit=20"
```

`200 OK` — a page of [Document](#document) objects (`last_job` is `null`).

#### Get a document

`GET /v1/datasets/{dataset_id}/documents/{document_id}`

Includes the most recent job in `last_job`, which is handy for tracking indexing progress per document.

```bash
curl -s $B/datasets/52967db904d44a09b33f88b3c9f64a6a/documents/bbd713912b614848a8f0def3f73aa304
```

`200 OK`

```json
{
  "id": "bbd713912b614848a8f0def3f73aa304",
  "dataset_id": "52967db904d44a09b33f88b3c9f64a6a",
  "filename": "contract.pdf",
  "mime_type": "application/pdf",
  "size_bytes": 3033,
  "sha256": "3c99bdb29e8d2ba87e7e7ff4d88664786e6128498e2e17327ac4ea727bb0c70c",
  "status": "indexed",
  "parser": "docling",
  "parser_backend": null,
  "num_chunks": 2,
  "indexed_at": "2026-11-04T19:31:51.340191Z",
  "last_job_id": "2d18380d5e6548ed868383387a606e3b",
  "metadata": {"year": 2025, "category": "contract"},
  "created_at": "2026-11-04T19:31:49.280215Z",
  "updated_at": "2026-11-04T19:31:51.340191Z",
  "last_job": {
    "id": "2d18380d5e6548ed868383387a606e3b",
    "dataset_id": "52967db904d44a09b33f88b3c9f64a6a",
    "document_id": "bbd713912b614848a8f0def3f73aa304",
    "parser": "docling",
    "options": {"do_ocr": false},
    "force_reparse": false,
    "status": "succeeded",
    "stage": "storing",
    "progress": 1.0,
    "error": null,
    "created_at": "2026-11-04T19:31:49.285116Z",
    "started_at": "2026-11-04T19:31:49.293914Z",
    "finished_at": "2026-11-04T19:31:51.340191Z"
  }
}
```

Errors: `404 dataset_not_found`, `404 document_not_found` (also when the document belongs to another dataset).

#### Update document metadata

`PATCH /v1/datasets/{dataset_id}/documents/{document_id}`

Replaces the document's metadata. The change applies to already indexed chunks immediately, so search filters see it without re-indexing.

```bash
curl -s -X PATCH $B/datasets/52967db904d44a09b33f88b3c9f64a6a/documents/bbd713912b614848a8f0def3f73aa304 \
  -H 'content-type: application/json' \
  -d '{"metadata": {"year": 2025, "category": "contract", "reviewed": true}}'
```

`200 OK` — the updated [Document](#document).

Errors: `422 invalid_metadata`.

#### Delete a document

`DELETE /v1/datasets/{dataset_id}/documents/{document_id}`

Cancels an active job, then deletes the document's chunks, its record and jobs, the stored file, and the parse cache.

```bash
curl -s -X DELETE $B/datasets/52967db904d44a09b33f88b3c9f64a6a/documents/bbd713912b614848a8f0def3f73aa304
```

`204 No Content`

#### Download the original file

`GET /v1/datasets/{dataset_id}/documents/{document_id}/content`

Returns the file as uploaded, with its original `Content-Type` and `Content-Disposition: attachment; filename="..."`.

```bash
curl -s -OJ $B/datasets/52967db904d44a09b33f88b3c9f64a6a/documents/bbd713912b614848a8f0def3f73aa304/content
```

Errors: `404 content_missing` if the stored file is gone.

#### Get the parsed markdown

`GET /v1/datasets/{dataset_id}/documents/{document_id}/parsed`

Returns the markdown produced by the parser for the most recent successful job (`text/markdown`). Useful to check parser quality.

```bash
curl -s $B/datasets/52967db904d44a09b33f88b3c9f64a6a/documents/bbd713912b614848a8f0def3f73aa304/parsed
```

`200 OK`

```markdown
## Service Agreement

Article 1. Purpose

This agreement defines the terms of the cloud storage service.

## Article 5. Termination

Either party may terminate with 30 days written notice.
```

Errors: `404 not_parsed` if the document was never indexed successfully.

---

### Indexing and jobs

Indexing runs in the background: parse → chunk → embed → store. The request returns a job right away; poll the job (or the document) to follow progress.

Re-indexing an indexed document replaces its chunks without a gap: new chunks are written first, then the old ones are removed. If the new job fails or is canceled, the old chunks stay.

Parse results are cached per parser and options. Re-indexing with the same parser and options skips parsing unless `force_reparse` is `true`.

#### Index a document

`POST /v1/datasets/{dataset_id}/documents/{document_id}/index`

Headers: `X-Model-API-Key` if required.

Request body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `parser` | string | `docling` | `docling` or `mineru` (`.md`/`.txt` always use `passthrough`). |
| `options` | object | `{}` | [Parser options](#parser-options). |
| `force_reparse` | boolean | `false` | Ignore the parse cache. |

Re-index an existing document with MinerU:

```bash
curl -s -X POST $B/datasets/52967db904d44a09b33f88b3c9f64a6a/documents/bbd713912b614848a8f0def3f73aa304/index \
  -H 'content-type: application/json' \
  -d '{"parser": "mineru", "options": {"tier": "basic"}}'
```

`202 Accepted` — a [Job](#job).

```json
{
  "id": "12364c983d5049e297af38072fb9f2b6",
  "dataset_id": "52967db904d44a09b33f88b3c9f64a6a",
  "document_id": "bbd713912b614848a8f0def3f73aa304",
  "parser": "mineru",
  "options": {"tier": "basic"},
  "force_reparse": false,
  "status": "queued",
  "stage": null,
  "progress": 0.0,
  "error": null,
  "created_at": "2026-11-04T19:31:51.463632Z",
  "started_at": null,
  "finished_at": null
}
```

Errors:

| Status | Code | When |
|--------|------|------|
| 409 | `job_in_progress` | The document already has a queued or running job. |
| 422 | `unsupported_parser_or_format` | Unknown parser, or the parser does not support the file type. |
| 422 | `invalid_parser_options` | Unknown option or wrong value type. |
| 400 | `missing_model_api_key` | The key is required but missing. |
| 409 | `embedding_model_unavailable` | The dataset's embedding model was removed from the catalog. |
| 503 | `parser_unavailable` | The parser service is down. |

```json
{"error": {"code": "job_in_progress", "message": "document 'bbd713912b614848a8f0def3f73aa304' already has an active indexing job"}}
```

```json
{"error": {"code": "invalid_parser_options", "message": "unsupported docling options: ['to_formats'] (allowed: ['do_ocr', 'do_table_structure', 'document_timeout', 'force_ocr', 'ocr_engine', 'ocr_lang', 'pdf_backend', 'table_mode'])"}}
```

#### Index documents in bulk

`POST /v1/datasets/{dataset_id}/index`

Creates one job per eligible document. Takes the same body as [Index a document](#index-a-document), plus:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scope` | string | `unindexed` | `unindexed`: documents with status `uploaded` or `failed`. `all`: every document (re-index). |

Documents that cannot be indexed are listed in `skipped` with a reason (an error code such as `job_in_progress` or `unsupported_parser_or_format`). The whole request fails instead if the options are invalid or the parser is down.

```bash
curl -s -X POST $B/datasets/52967db904d44a09b33f88b3c9f64a6a/index \
  -H 'content-type: application/json' \
  -d '{"parser": "docling", "scope": "unindexed"}'
```

`202 Accepted`

```json
{
  "jobs": [
    {
      "id": "7f370c764949432f86adad232a700b05",
      "dataset_id": "52967db904d44a09b33f88b3c9f64a6a",
      "document_id": "d67f7ae3fe2a43f488a479d72c1b4ad9",
      "parser": "passthrough",
      "options": {},
      "force_reparse": false,
      "status": "queued",
      "stage": null,
      "progress": 0.0,
      "error": null,
      "created_at": "2026-11-04T19:31:51.511038Z",
      "started_at": null,
      "finished_at": null
    }
  ],
  "skipped": [
    {"document_id": "0a1b2c3d4e5f60718293a4b5c6d7e8f9", "reason": "unsupported_parser_or_format"}
  ]
}
```

#### List jobs

`GET /v1/datasets/{dataset_id}/jobs`

| Query | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | string | | `queued`, `running`, `succeeded`, `failed`, or `canceled`. |
| `document_id` | string | | Jobs of one document (its indexing history). |
| `limit` | integer | `50` | 1–500. |
| `offset` | integer | `0` | |

```bash
curl -s "$B/datasets/52967db904d44a09b33f88b3c9f64a6a/jobs?status=running"
curl -s "$B/datasets/52967db904d44a09b33f88b3c9f64a6a/jobs?document_id=bbd713912b614848a8f0def3f73aa304"
```

`200 OK` — a page of [Job](#job) objects.

#### Get a job

`GET /v1/datasets/{dataset_id}/jobs/{job_id}`

```bash
curl -s $B/datasets/52967db904d44a09b33f88b3c9f64a6a/jobs/2d18380d5e6548ed868383387a606e3b
```

`200 OK`

```json
{
  "id": "2d18380d5e6548ed868383387a606e3b",
  "dataset_id": "52967db904d44a09b33f88b3c9f64a6a",
  "document_id": "bbd713912b614848a8f0def3f73aa304",
  "parser": "docling",
  "options": {"do_ocr": false},
  "force_reparse": false,
  "status": "succeeded",
  "stage": "storing",
  "progress": 1.0,
  "error": null,
  "created_at": "2026-11-04T19:31:49.285116Z",
  "started_at": "2026-11-04T19:31:49.293914Z",
  "finished_at": "2026-11-04T19:31:51.340191Z"
}
```

A failed job explains why in `error`, for example:

```json
{"status": "failed", "stage": "parsing", "error": "mineru parsing failed: corrupted pdf", "...": "..."}
```

Polling loop:

```bash
JOB=2d18380d5e6548ed868383387a606e3b
until curl -s $B/datasets/$DS/jobs/$JOB | jq -e '.status | IN("succeeded","failed","canceled")' >/dev/null; do
  curl -s $B/datasets/$DS/jobs/$JOB | jq -c '{status, stage, progress}'
  sleep 2
done
```

#### Cancel a job

`POST /v1/datasets/{dataset_id}/jobs/{job_id}/cancel`

A queued job is canceled immediately. A running job is interrupted, its partial chunks are removed, and the call returns once cleanup is done. The remote parse task (docling/MinerU) is canceled on a best-effort basis.

```bash
curl -s -X POST $B/datasets/52967db904d44a09b33f88b3c9f64a6a/jobs/12364c983d5049e297af38072fb9f2b6/cancel
```

`200 OK` — the [Job](#job) with `status: "canceled"`.

Errors: `409 job_not_active` if the job already finished.

```json
{"error": {"code": "job_not_active", "message": "job is already succeeded"}}
```

---

### Search

#### Search a dataset

`POST /v1/datasets/{dataset_id}/search`

Returns the chunks most similar to the query, from all indexed documents of one dataset.

Headers: `X-Model-API-Key` if required.

Request body:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | | 1–8000 characters. |
| `top_k` | integer | no | `5` | 1 to `MAX_TOP_K` (default 50). |
| `filters.document_ids` | string[] | no | | Only search these documents. An empty list returns no results. |
| `filters.metadata` | object | no | | Exact match on document metadata. All pairs must match, and types must match too: `{"year": 2025}` does not match the string `"2025"`. Keys may also be `parser`, `page`, `filename`, `chunk_index`. |

```bash
curl -s -X POST $B/datasets/52967db904d44a09b33f88b3c9f64a6a/search \
  -H 'content-type: application/json' \
  -d '{
    "query": "how can the agreement be terminated",
    "top_k": 2,
    "filters": {"metadata": {"year": 2025}}
  }'
```

`200 OK`

```json
{
  "query": "how can the agreement be terminated",
  "results": [
    {
      "score": 0.694,
      "text": "## Article 5. Termination\n\nEither party may terminate with 30 days written notice.",
      "document_id": "bbd713912b614848a8f0def3f73aa304",
      "filename": "contract.pdf",
      "page": 2,
      "section_path": ["Article 5. Termination"],
      "chunk_index": 1,
      "metadata": {"year": 2025, "category": "contract"}
    },
    {
      "score": 0.2,
      "text": "## Service Agreement\n\nArticle 1. Purpose\n\nThis agreement defines the terms of the cloud storage service.",
      "document_id": "bbd713912b614848a8f0def3f73aa304",
      "filename": "contract.pdf",
      "page": 1,
      "section_path": ["Service Agreement"],
      "chunk_index": 0,
      "metadata": {"year": 2025, "category": "contract"}
    }
  ]
}
```

Only search two documents:

```bash
curl -s -X POST $B/datasets/52967db904d44a09b33f88b3c9f64a6a/search \
  -H 'content-type: application/json' \
  -d '{"query": "termination notice period", "filters": {"document_ids": ["bbd713912b614848a8f0def3f73aa304", "d67f7ae3fe2a43f488a479d72c1b4ad9"]}}'
```

A dataset with no indexed documents returns `200` with `"results": []`.

Errors: `422 validation_error`, `422 top_k_too_large`, `422 invalid_filter` (malformed document id or metadata key), `400 missing_model_api_key`.

```json
{"error": {"code": "top_k_too_large", "message": "top_k must be <= 50"}}
```

> Searching several datasets in one request is not supported. Search each dataset separately.

---

### Parsers

#### List parsers

`GET /v1/parsers`

Checks each parser service and reports whether it is reachable. Use it to decide which `parser` to request.

```bash
curl -s $B/parsers
```

`200 OK`

```json
[
  {
    "name": "passthrough",
    "available": true,
    "detail": null,
    "version": null,
    "supported_extensions": [".markdown", ".md", ".txt"]
  },
  {
    "name": "docling",
    "available": true,
    "detail": null,
    "version": "1.38.2",
    "supported_extensions": [".adoc", ".asciidoc", ".bmp", ".csv", ".docx", ".htm", ".html", ".jpeg", ".jpg", ".pdf", ".png", ".pptx", ".tif", ".tiff", ".webp", ".xhtml", ".xlsx"]
  },
  {
    "name": "mineru",
    "available": true,
    "detail": "tiers: flash, basic; default tier: basic",
    "version": "4.1.7",
    "supported_extensions": [".bmp", ".csv", ".doc", ".docx", ".epub", ".gif", ".htm", ".html", ".jp2", ".jpeg", ".jpg", ".mht", ".mhtml", ".odp", ".ods", ".odt", ".pdf", ".png", ".ppt", ".pptx", ".rtf", ".tif", ".tiff", ".tsv", ".webp", ".xls", ".xlsx"]
  }
]
```

---

### Operations

#### Liveness

`GET /healthz` → `200 {"status": "ok"}`

#### Readiness

`GET /readyz` → `200 {"status": "ready"}` when the database is reachable, otherwise `503 db_unavailable`.

Parser availability is not part of readiness (search keeps working when a parser is down); check `GET /v1/parsers` instead.

---

## Parser options

Pass these in `options` when indexing. Unknown options are rejected with `422 invalid_parser_options`.

### docling

| Option | Type | Description |
|--------|------|-------------|
| `do_ocr` | boolean | Run OCR on bitmap content. Default on. Turn off for born-digital PDFs to speed things up. |
| `force_ocr` | boolean | OCR every page, replacing the text layer. |
| `ocr_engine` | string | OCR engine, e.g. `easyocr`, `tesseract`, `rapidocr`. |
| `ocr_lang` | string[] | OCR languages, e.g. `["ko", "en"]` for Korean scans. |
| `do_table_structure` | boolean | Recognize table structure. Default on. |
| `table_mode` | string | `fast` or `accurate`. |
| `pdf_backend` | string | PDF backend, e.g. `docling_parse`, `pypdfium2`. |
| `document_timeout` | number | Per-document timeout in seconds on the docling side. |

```json
{"parser": "docling", "options": {"do_ocr": true, "ocr_lang": ["ko", "en"], "table_mode": "accurate"}}
```

### mineru

| Option | Type | Description |
|--------|------|-------------|
| `tier` | string | `flash` (text layer only), `basic` (lightweight local models, CPU), `standard` / `advanced` (VLM; requires a GPU deployment). Defaults to the server's `MINERU_TIER` (`basic` on CPU). Applies to PDFs and images only; other formats always use `flash`. |
| `ocr_mode` | string | `auto`, `txt` (use the text layer), or `ocr`. Default `auto`. |
| `page_range` | string | PDF pages to parse, 1-based, e.g. `"1-5,8"`; `r1` is the last page. |

```json
{"parser": "mineru", "options": {"tier": "basic", "ocr_mode": "auto", "page_range": "1-20"}}
```

The tier actually used is recorded in the job's `options` and in the document's `parser_backend` (e.g. `tier:basic`). Requesting a tier the MinerU server does not offer fails the job with an error from MinerU.

### passthrough

Used automatically for `.md`, `.markdown`, `.txt`. Takes no options.

---

## Error codes

| Status | Code | Meaning |
|--------|------|---------|
| 400 | `missing_model_api_key` | `X-Model-API-Key` is required for this dataset's embedding model. |
| 404 | `dataset_not_found` | No such dataset (or malformed id). |
| 404 | `document_not_found` | No such document in this dataset. |
| 404 | `job_not_found` | No such job in this dataset. |
| 404 | `content_missing` | The stored original file is missing. |
| 404 | `not_parsed` | The document has no successful parse yet. |
| 404 | `not_found` | Unknown route. |
| 405 | `method_not_allowed` | Wrong HTTP method for the route. |
| 409 | `external_ref_conflict` | `external_ref` already used by another dataset. |
| 409 | `duplicate_document` | Same file content already in the dataset. |
| 409 | `job_in_progress` | The document already has an active job. |
| 409 | `job_not_active` | The job to cancel has already finished. |
| 409 | `embedding_model_unavailable` | The dataset's embedding model is no longer in the catalog. |
| 413 | `file_too_large` | Upload exceeds `MAX_UPLOAD_MB`. |
| 422 | `validation_error` | Request body or query failed schema validation. |
| 422 | `unknown_embedding_model` | `embedding_model_id` is not in the catalog. |
| 422 | `chunk_size_too_large` | `chunk_size` exceeds the model's `max_input_tokens`. |
| 422 | `invalid_chunk_overlap` | `chunk_overlap` is not less than `chunk_size`. |
| 422 | `invalid_metadata` | Bad metadata key or value, or a reserved key. |
| 422 | `invalid_filename` | Upload has no usable file name. |
| 422 | `invalid_json` | A JSON form field (`metadata`, `options`) is malformed or not an object. |
| 422 | `unsupported_parser_or_format` | Unknown parser, or the parser cannot handle the file type. |
| 422 | `invalid_parser_options` | Unknown parser option or wrong value type. |
| 422 | `invalid_filter` | Malformed `document_ids` or metadata filter key. |
| 422 | `top_k_too_large` | `top_k` exceeds `MAX_TOP_K`. |
| 503 | `parser_unavailable` | The requested parser service is unreachable. |
| 503 | `db_unavailable` | The database is unreachable (`/readyz`). |

---

## End-to-end example

```bash
B=http://localhost:8000/v1
J='content-type: application/json'
# KEY="sk-..."   # needed once a LiteLLM-backed embedding model is used: add -H "X-Model-API-Key: $KEY"

# 1. Pick an embedding model and create a dataset
curl -s $B/embedding-models | jq '.[].id'
DS=$(curl -s -X POST $B/datasets -H "$J" \
  -d '{"name": "legal-docs", "embedding_model_id": "hash-dev", "external_ref": "team-legal-001"}' | jq -r .id)

# 2. Check which parsers are up
curl -s $B/parsers | jq -c '.[] | {name, available, version}'

# 3. Upload a PDF and index it with MinerU
JOB=$(curl -s -X POST $B/datasets/$DS/documents \
  -F file=@contract.pdf -F index=true -F parser=mineru \
  -F 'metadata={"year": 2025}' | jq -r .job.id)

# 4. Upload more files without indexing, then index them all with docling
curl -s -X POST $B/datasets/$DS/documents -F file=@policy.docx >/dev/null
curl -s -X POST $B/datasets/$DS/documents -F file=@faq.md >/dev/null
curl -s -X POST $B/datasets/$DS/index -H "$J" -d '{"parser": "docling"}' | jq '{jobs: [.jobs[].id], skipped}'

# 5. Wait for the MinerU job
until curl -s $B/datasets/$DS/jobs/$JOB | jq -e '.status | IN("succeeded","failed","canceled")' >/dev/null; do sleep 2; done
curl -s $B/datasets/$DS/jobs/$JOB | jq -c '{status, progress, error}'

# 6. Overview of the dataset
curl -s $B/datasets/$DS | jq .document_counts

# 7. Search
curl -s -X POST $B/datasets/$DS/search -H "$J" \
  -d '{"query": "how can the agreement be terminated", "top_k": 3}' \
  | jq -c '.results[] | {score, filename, page, section_path}'

# 8. Clean up
curl -s -X DELETE $B/datasets/$DS
```
