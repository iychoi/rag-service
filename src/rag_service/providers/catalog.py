"""Embedding model catalog.

Every dataset must pick one of these ids at creation time, because the dimension must be known
up front to create the dataset's vector table (`vector(dim)`).
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Maximum dimension for an HNSW index on pgvector's `vector` type
PGVECTOR_HNSW_MAX_DIM = 2000


class InputTypes(BaseModel):
    """Per-document/query input_type for asymmetric embedding models (e.g. Cohere)."""

    document: str
    query: str


class EmbeddingModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    provider: str
    model: str
    dim: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    input_type: InputTypes | None = None
    description: str | None = None

    @field_validator("dim")
    @classmethod
    def _dim_limit(cls, v: int) -> int:
        if v > PGVECTOR_HNSW_MAX_DIM:
            raise ValueError(
                f"dim {v} exceeds pgvector HNSW limit ({PGVECTOR_HNSW_MAX_DIM}); "
                "use a reduced output dimension"
            )
        return v


class EmbeddingCatalog:
    def __init__(self, entries: list[EmbeddingModelEntry]) -> None:
        ids = [e.id for e in entries]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate embedding model ids in catalog: {sorted(dupes)}")
        self._entries = {e.id: e for e in entries}

    @classmethod
    def load(cls, path: Path) -> "EmbeddingCatalog":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ValueError(f"{path}: catalog must be a YAML list")
        return cls([EmbeddingModelEntry.model_validate(item) for item in raw])

    def get(self, model_id: str) -> EmbeddingModelEntry | None:
        return self._entries.get(model_id)

    def all(self) -> list[EmbeddingModelEntry]:
        return list(self._entries.values())
