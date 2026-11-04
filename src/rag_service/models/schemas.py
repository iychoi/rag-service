"""API request/response schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_service.models.tables import Dataset, Document, DocumentStatus, IndexJob, JobStage, JobStatus


DS_ID = "52967db904d44a09b33f88b3c9f64a6a"
DOC_ID = "bbd713912b614848a8f0def3f73aa304"
JOB_ID = "2d18380d5e6548ed868383387a606e3b"

JOB_EXAMPLE: dict[str, Any] = {
    "id": JOB_ID,
    "dataset_id": DS_ID,
    "document_id": DOC_ID,
    "parser": "docling",
    "options": {"do_ocr": False},
    "force_reparse": False,
    "status": "succeeded",
    "stage": "storing",
    "progress": 1.0,
    "error": None,
    "created_at": "2026-11-04T19:31:49.285116Z",
    "started_at": "2026-11-04T19:31:49.293914Z",
    "finished_at": "2026-11-04T19:31:51.340191Z",
}

DOCUMENT_EXAMPLE: dict[str, Any] = {
    "id": DOC_ID,
    "dataset_id": DS_ID,
    "filename": "contract.pdf",
    "mime_type": "application/pdf",
    "size_bytes": 3033,
    "sha256": "3c99bdb29e8d2ba87e7e7ff4d88664786e6128498e2e17327ac4ea727bb0c70c",
    "status": "indexed",
    "parser": "docling",
    "parser_backend": None,
    "num_chunks": 2,
    "indexed_at": "2026-11-04T19:31:51.340191Z",
    "last_job_id": JOB_ID,
    "metadata": {"year": 2025, "category": "contract"},
    "created_at": "2026-11-04T19:31:49.280215Z",
    "updated_at": "2026-11-04T19:31:51.340191Z",
    "last_job": JOB_EXAMPLE,
}


class ErrorBody(BaseModel):
    code: str = Field(description="Stable, machine-readable error code")
    message: str = Field(description="Human-readable message")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"error": {"code": "dataset_not_found", "message": f"dataset '{DS_ID}' not found"}}]
        }
    )

    error: ErrorBody


class Page[T](BaseModel):
    items: list[T]
    total: int = Field(description="Total number of matching items")
    limit: int
    offset: int


# ---------- embedding models ----------


class InputTypes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: str = Field(min_length=1, max_length=64, description="input_type used when indexing")
    query: str = Field(min_length=1, max_length=64, description="input_type used for search queries")


class EmbeddingConfig(BaseModel):
    """Embedding model of a dataset, fixed at creation."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["litellm", "hash"] = Field(
        default="litellm",
        description="litellm: a model registered in the server's LiteLLM proxy (needs "
        "X-Model-API-Key). hash: local lexical embedding for development/tests only.",
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="LiteLLM model name, e.g. titan-embed-v2. Required for litellm; ignored for hash.",
    )
    dimensions: int | None = Field(
        default=None,
        ge=1,
        le=2000,
        description="Requested output dimension, for models that support it (e.g. Titan V2: 256/512/1024). "
        "For hash: the vector size (default 512).",
    )
    input_type: InputTypes | None = Field(
        default=None,
        description="For asymmetric models (e.g. Cohere): input_type sent for documents and queries, "
        'e.g. {"document": "search_document", "query": "search_query"}',
    )


class EmbeddingOut(BaseModel):
    provider: str
    model: str
    dim: int = Field(description="Embedding dimension (measured at dataset creation)")
    dimensions: int | None = Field(default=None, description="Requested output dimension, if any")
    input_type: InputTypes | None = None


# ---------- datasets ----------


class DatasetCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "name": "legal-docs",
                    "embedding": {"provider": "litellm", "model": "titan-embed-v2"},
                    "external_ref": "team-legal-001",
                    "description": "Contracts and policies",
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "metadata": {"team": "legal"},
                }
            ]
        },
    )

    name: str = Field(min_length=1, max_length=255)
    embedding: EmbeddingConfig = Field(
        description="Embedding model. Cannot be changed later. The base URL is server configuration."
    )
    external_ref: str | None = Field(
        default=None, max_length=255, description="Unique identifier from an external system"
    )
    description: str | None = None
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Tokens per chunk; must fit the embedding model's input limit (indexing fails otherwise)",
    )
    chunk_overlap: int = Field(default=64, ge=0, le=2048, description="Must be less than chunk_size")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Keys: ^[A-Za-z][A-Za-z0-9_.-]{0,63}$; scalar values"
    )


class DatasetUpdate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"description": "Contracts, policies, and amendments"}]},
    )

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    metadata: dict[str, Any] | None = None


class DatasetOut(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": DS_ID,
                    "name": "legal-docs",
                    "external_ref": "team-legal-001",
                    "description": "Contracts and policies",
                    "embedding": {
                        "provider": "litellm",
                        "model": "titan-embed-v2",
                        "dim": 1024,
                        "dimensions": None,
                        "input_type": None,
                    },
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "metadata": {"team": "legal"},
                    "created_at": "2026-11-04T19:31:49.174949Z",
                    "updated_at": "2026-11-04T19:31:49.174960Z",
                    "document_counts": {"indexed": 1, "uploaded": 1, "total": 2},
                }
            ]
        }
    )

    id: str
    name: str
    external_ref: str | None
    description: str | None
    embedding: EmbeddingOut
    chunk_size: int
    chunk_overlap: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    document_counts: dict[str, int] | None = Field(
        default=None, description="Documents per status plus `total`; null in list responses"
    )

    @classmethod
    def from_row(cls, ds: Dataset, counts: dict[str, int] | None = None) -> "DatasetOut":
        return cls(
            id=ds.id,
            name=ds.name,
            external_ref=ds.external_ref,
            description=ds.description,
            embedding=EmbeddingOut(
                provider=ds.embedding_provider,
                model=ds.embedding_model,
                dim=ds.embedding_dim,
                dimensions=ds.embedding_options.get("dimensions"),
                input_type=ds.embedding_options.get("input_type"),
            ),
            chunk_size=ds.chunk_size,
            chunk_overlap=ds.chunk_overlap,
            metadata=ds.meta,
            created_at=ds.created_at,
            updated_at=ds.updated_at,
            document_counts=counts,
        )


# ---------- jobs ----------


class JobOut(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [JOB_EXAMPLE]})

    id: str
    dataset_id: str
    document_id: str
    parser: str = Field(description="Parser actually used: docling, mineru, or passthrough")
    options: dict[str, Any] = Field(description="Effective parser options (MinerU always includes tier)")
    force_reparse: bool
    status: JobStatus
    stage: JobStage | None = Field(description="Last stage reached")
    progress: float = Field(description="0.0–1.0 (parse ≤0.4, chunk ≤0.45, embed ≤0.9, store ≤1.0)")
    error: str | None = Field(description="Failure reason for failed/canceled jobs")
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def from_row(cls, job: IndexJob) -> "JobOut":
        return cls.model_validate(job, from_attributes=True)


class IndexRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"parser": "mineru", "options": {"tier": "basic"}}]},
    )

    parser: str = Field(
        default="docling",
        description="docling or mineru (.md/.markdown/.txt always use passthrough)",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Parser options. docling: do_ocr, force_ocr, ocr_engine, ocr_lang, "
        "do_table_structure, table_mode, pdf_backend, document_timeout. "
        "mineru: tier (flash|basic|standard|advanced), ocr_mode (auto|txt|ocr), page_range.",
    )
    force_reparse: bool = Field(default=False, description="Ignore the parse cache")


class BulkIndexRequest(IndexRequest):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"parser": "docling", "scope": "unindexed"}]},
    )

    scope: Literal["unindexed", "all"] = Field(
        default="unindexed",
        description="unindexed: documents with status uploaded or failed; all: every document",
    )


class BulkIndexOut(BaseModel):
    jobs: list[JobOut]
    skipped: list[dict[str, str]] = Field(
        description="Documents not indexed, as {document_id, reason} where reason is an error code"
    )


# ---------- documents ----------


class DocumentOut(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [DOCUMENT_EXAMPLE]})

    id: str
    dataset_id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    status: DocumentStatus
    parser: str | None = Field(description="Parser that produced the current index")
    parser_backend: str | None = Field(description="Parser backend detail, e.g. tier:basic")
    num_chunks: int | None
    indexed_at: datetime | None
    last_job_id: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_job: JobOut | None = Field(
        default=None, description="Most recent job; null in list responses"
    )

    @classmethod
    def from_row(cls, doc: Document, last_job: IndexJob | None = None) -> "DocumentOut":
        return cls(
            id=doc.id,
            dataset_id=doc.dataset_id,
            filename=doc.filename,
            mime_type=doc.mime_type,
            size_bytes=doc.size_bytes,
            sha256=doc.sha256,
            status=doc.status,
            parser=doc.parser,
            parser_backend=doc.parser_backend,
            num_chunks=doc.num_chunks,
            indexed_at=doc.indexed_at,
            last_job_id=doc.last_job_id,
            metadata=doc.meta,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            last_job=JobOut.from_row(last_job) if last_job else None,
        )


class DocumentUploadOut(BaseModel):
    document: DocumentOut
    job: JobOut | None = Field(default=None, description="Indexing job when index=true, else null")


class DocumentUpdate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"metadata": {"year": 2025, "category": "contract", "reviewed": True}}]
        },
    )

    metadata: dict[str, Any] = Field(description="Replaces the whole metadata object")


# ---------- search ----------


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: list[str] | None = Field(
        default=None, description="Only search these documents; an empty list returns no results"
    )
    metadata: dict[str, str | int | float | bool] | None = Field(
        default=None, description="Exact match on document metadata; types must match too"
    )


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "query": "how can the agreement be terminated",
                    "top_k": 2,
                    "filters": {"metadata": {"year": 2025}},
                }
            ]
        },
    )

    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, description="At most MAX_TOP_K (default 50)")
    filters: SearchFilters | None = None


class SearchResult(BaseModel):
    score: float = Field(description="Cosine similarity (higher is closer)")
    text: str = Field(description="Chunk text (markdown)")
    document_id: str
    filename: str | None
    page: int | None = Field(description="1-based page number when the parser reports pages")
    section_path: list[str] = Field(description="Heading path of the chunk")
    chunk_index: int | None
    metadata: dict[str, Any] = Field(description="Current document metadata")


class SearchResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "how can the agreement be terminated",
                    "results": [
                        {
                            "score": 0.694,
                            "text": "## Article 5. Termination\n\nEither party may terminate with 30 days written notice.",
                            "document_id": DOC_ID,
                            "filename": "contract.pdf",
                            "page": 2,
                            "section_path": ["Article 5. Termination"],
                            "chunk_index": 1,
                            "metadata": {"year": 2025, "category": "contract"},
                        }
                    ],
                }
            ]
        }
    )

    query: str
    results: list[SearchResult]


# ---------- parsers ----------


class ParserHealthOut(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "mineru",
                    "available": True,
                    "detail": "tiers: flash, basic; default tier: basic",
                    "version": "4.1.7",
                    "supported_extensions": [".docx", ".pdf", ".png"],
                }
            ]
        }
    )

    name: str
    available: bool
    detail: str | None = None
    version: str | None = None
    supported_extensions: list[str]


# ---------- query (retrieval + LLM answer) ----------


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "query": "How can the agreement be terminated?",
                    "top_k": 5,
                    "filters": {"metadata": {"year": 2025}},
                    "model": "claude-sonnet-5-5",
                    "max_output_tokens": 1024,
                }
            ]
        },
    )

    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(
        default=5, ge=1, description="Chunks retrieved as sources; at most MAX_TOP_K (default 50)"
    )
    filters: SearchFilters | None = None
    model: str | None = Field(
        default=None,
        description="LiteLLM model that writes the answer, passed to LiteLLM as-is. Omit to use the "
        "server default (LLM_MODEL, default `claude-sonnet-5-5`). A model LiteLLM does not know "
        "returns its error as 502 upstream_error.",
        examples=["claude-sonnet-5-5", "claude-haiku-5-5"],
    )
    max_output_tokens: int | None = Field(
        default=None, ge=1, le=8192, description="Answer length limit. Default: LLM_MAX_OUTPUT_TOKENS"
    )


class QuerySource(SearchResult):
    id: int = Field(description="Reference number used for citations in the answer, e.g. [1]")


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "How can the agreement be terminated?",
                    "answer": "Either party may terminate the agreement with 30 days written notice [1].",
                    "model": "claude-sonnet-5-5",
                    "citations": [1],
                    "sources": [
                        {
                            "id": 1,
                            "score": 0.694,
                            "text": "## Article 5. Termination\n\nEither party may terminate with 30 days written notice.",
                            "document_id": DOC_ID,
                            "filename": "contract.pdf",
                            "page": 2,
                            "section_path": ["Article 5. Termination"],
                            "chunk_index": 1,
                            "metadata": {"year": 2025, "category": "contract"},
                        }
                    ],
                    "usage": {"input_tokens": 412, "output_tokens": 23},
                }
            ]
        }
    )

    query: str
    answer: str | None = Field(
        description="Answer grounded in the sources, with [n] citations. "
        "null when no sources were found (the LLM is not called)."
    )
    model: str | None = Field(description="Model that produced the answer; null if not called")
    citations: list[int] = Field(description="Source ids cited in the answer, in order of first use")
    sources: list[QuerySource] = Field(description="Retrieved chunks given to the LLM")
    usage: TokenUsage | None = Field(default=None, description="Token usage reported by LiteLLM")
