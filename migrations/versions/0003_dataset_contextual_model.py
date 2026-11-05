"""datasets can generate a context for each chunk with an LLM (contextual retrieval)

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("contextual_model", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("datasets", "contextual_model")
