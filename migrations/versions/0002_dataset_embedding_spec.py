"""datasets choose their embedding model directly (no server-side catalog)

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column("embedding_options", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    # Existing datasets were created from the catalog; keep their dimension as an explicit option
    # for the hash provider so the same vectors are produced.
    op.execute(
        "UPDATE datasets SET embedding_options = jsonb_build_object('dimensions', embedding_dim) "
        "WHERE embedding_provider = 'hash'"
    )
    op.drop_column("datasets", "embedding_model_id")


def downgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column("embedding_model_id", sa.String(128), nullable=False, server_default="hash-dev"),
    )
    op.drop_column("datasets", "embedding_options")
