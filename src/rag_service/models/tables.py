"""SQLModel table definitions.

`metadata` is a reserved attribute on SQLAlchemy declarative models, so the Python attribute
is `meta` while the DB column is named `metadata`.
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Column, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentStatus(StrEnum):
    uploaded = "uploaded"
    indexing = "indexing"
    indexed = "indexed"
    failed = "failed"


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"


class JobStage(StrEnum):
    parsing = "parsing"
    chunking = "chunking"
    embedding = "embedding"
    storing = "storing"


ACTIVE_JOB_STATUSES = (JobStatus.queued, JobStatus.running)


class Dataset(SQLModel, table=True):
    __tablename__ = "datasets"

    id: str = Field(default_factory=new_id, primary_key=True, max_length=32)
    name: str = Field(max_length=255)
    external_ref: str | None = Field(default=None, max_length=255, unique=True)
    description: str | None = None

    embedding_model_id: str = Field(max_length=128)
    embedding_provider: str = Field(max_length=64)
    embedding_model: str = Field(max_length=255)
    embedding_dim: int
    vector_table: str = Field(max_length=63)

    chunk_size: int
    chunk_overlap: int

    meta: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column("metadata", JSONB, nullable=False)
    )
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    updated_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class Document(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("dataset_id", "sha256", name="uq_documents_dataset_sha256"),)

    id: str = Field(default_factory=new_id, primary_key=True, max_length=32)
    dataset_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    filename: str = Field(max_length=1024)
    mime_type: str = Field(max_length=255)
    size_bytes: int
    sha256: str = Field(max_length=64)
    storage_path: str = Field(max_length=2048)

    status: DocumentStatus = Field(
        default=DocumentStatus.uploaded, sa_column=Column(String(16), nullable=False)
    )
    parser: str | None = Field(default=None, max_length=32)
    parser_backend: str | None = Field(default=None, max_length=64)
    num_chunks: int | None = None
    indexed_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    last_job_id: str | None = Field(default=None, max_length=32)

    meta: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column("metadata", JSONB, nullable=False)
    )
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    updated_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class IndexJob(SQLModel, table=True):
    __tablename__ = "index_jobs"

    id: str = Field(default_factory=new_id, primary_key=True, max_length=32)
    dataset_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    document_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    parser: str = Field(max_length=32)
    options: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    force_reparse: bool = False

    status: JobStatus = Field(
        default=JobStatus.queued, sa_column=Column(String(16), nullable=False, index=True)
    )
    stage: JobStage | None = Field(default=None, sa_column=Column(String(16), nullable=True))
    progress: float = 0.0
    error: str | None = None

    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    started_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    finished_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
