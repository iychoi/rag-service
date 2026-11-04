"""initial schema: metadata tables, pgvector extension, vectors schema

Revision ID: 0001
Revises:
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _ts(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE SCHEMA IF NOT EXISTS vectors")
    # Mitigate short HNSW result sets for filtered searches (pgvector >= 0.8)
    op.execute(
        "DO $$ BEGIN "
        "EXECUTE format('ALTER DATABASE %I SET hnsw.iterative_scan = relaxed_order', current_database()); "
        "END $$"
    )

    op.create_table(
        "datasets",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("external_ref", sa.String(255), nullable=True, unique=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("embedding_model_id", sa.String(128), nullable=False),
        sa.Column("embedding_provider", sa.String(64), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("embedding_dim", sa.Integer, nullable=False),
        sa.Column("vector_table", sa.String(63), nullable=False),
        sa.Column("chunk_size", sa.Integer, nullable=False),
        sa.Column("chunk_overlap", sa.Integer, nullable=False),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        _ts("created_at"),
        _ts("updated_at"),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "dataset_id",
            sa.String(32),
            sa.ForeignKey("datasets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(1024), nullable=False),
        sa.Column("mime_type", sa.String(255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("storage_path", sa.String(2048), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("parser", sa.String(32), nullable=True),
        sa.Column("parser_backend", sa.String(64), nullable=True),
        sa.Column("num_chunks", sa.Integer, nullable=True),
        _ts("indexed_at", nullable=True),
        sa.Column("last_job_id", sa.String(32), nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        _ts("created_at"),
        _ts("updated_at"),
        sa.UniqueConstraint("dataset_id", "sha256", name="uq_documents_dataset_sha256"),
    )
    op.create_index("ix_documents_dataset_id", "documents", ["dataset_id"])

    op.create_table(
        "index_jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "dataset_id",
            sa.String(32),
            sa.ForeignKey("datasets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("parser", sa.String(32), nullable=False),
        sa.Column("options", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("force_reparse", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("stage", sa.String(16), nullable=True),
        sa.Column("progress", sa.Float, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=True),
        _ts("created_at"),
        _ts("started_at", nullable=True),
        _ts("finished_at", nullable=True),
    )
    op.create_index("ix_index_jobs_dataset_id", "index_jobs", ["dataset_id"])
    op.create_index("ix_index_jobs_document_id", "index_jobs", ["document_id"])
    op.create_index("ix_index_jobs_status", "index_jobs", ["status"])
    # At most one active job per document (guards against concurrent requests)
    op.create_index(
        "uq_index_jobs_active_document",
        "index_jobs",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_table("index_jobs")
    op.drop_table("documents")
    op.drop_table("datasets")
    op.execute("DROP SCHEMA IF EXISTS vectors CASCADE")
