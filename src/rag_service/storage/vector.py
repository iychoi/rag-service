"""Per-dataset pgvector table management.

Uses one LlamaIndex PGVectorStore table per dataset.
- Table: vectors.data_ds_{dataset_id}  (PGVectorStore adds the `data_` prefix)
- Node ref_doc_id = document_id → per-document deletes use PGVectorStore's default btree index
- Node metadata `job_id` enables re-indexing as "insert new nodes, then delete old ones" (no gap)

Caution: PGVectorStore's MetadataFilters interpolate keys/values directly into SQL strings
(llama-index-vector-stores-postgres 0.9). Filters built from user input must never go through
that path; they are applied via `customize_query_fn` with hand-written SQLAlchemy expressions (bound parameters).
"""

import asyncio
import re
from collections import OrderedDict
from typing import Any

import sqlalchemy as sa
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from rag_service.db import Database
from rag_service.models.tables import Dataset

VECTOR_SCHEMA = "vectors"
_HNSW = {"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 64}
_SAFE_ID = re.compile(r"^[a-z0-9_]+$")


def vector_table_name(dataset_id: str) -> str:
    """table_name passed to PGVectorStore (the physical table gets a data_ prefix)."""
    name = f"ds_{dataset_id}"
    if not _SAFE_ID.match(name) or len(name) > 58:
        raise ValueError(f"invalid dataset id for table name: {dataset_id!r}")
    return name


def _physical_table(ds: Dataset) -> sa.TableClause:
    return sa.table(
        f"data_{ds.vector_table}", sa.column("metadata_", JSONB), schema=VECTOR_SCHEMA
    )


def _apply_user_filters(
    stmt: sa.Select,
    table_class: Any,
    rag_document_ids: list[str] | None = None,
    rag_metadata: dict[str, Any] | None = None,
    **_: Any,
) -> sa.Select:
    """Apply search filters with bound parameters (PGVectorStore customize_query_fn)."""
    if rag_document_ids:
        stmt = stmt.where(table_class.metadata_["ref_doc_id"].astext.in_(rag_document_ids))
    if rag_metadata:
        # JSONB containment (@>) compares types exactly: {"year": 2025} only matches the number 2025
        stmt = stmt.where(table_class.metadata_.contains(rag_metadata))
    return stmt


class VectorStoreManager:
    def __init__(self, db: Database, cache_size: int = 256) -> None:
        self._db = db
        self._cache: OrderedDict[str, PGVectorStore] = OrderedDict()
        self._cache_size = cache_size

    def _new_store(self, ds: Dataset, perform_setup: bool) -> PGVectorStore:
        return PGVectorStore(
            connection_string=self._db.url,
            async_connection_string=self._db.url,
            engine=self._db.sync_engine,
            async_engine=self._db.async_engine,
            schema_name=VECTOR_SCHEMA,
            table_name=ds.vector_table,
            embed_dim=ds.embedding_dim,
            use_jsonb=True,
            hnsw_kwargs=dict(_HNSW, hnsw_dist_method="vector_cosine_ops"),
            perform_setup=perform_setup,
            initialization_fail_on_error=True,
            customize_query_fn=_apply_user_filters,
        )

    def _store(self, ds: Dataset) -> PGVectorStore:
        store = self._cache.get(ds.id)
        if store is None:
            # Tables are created with the dataset, so only connect here (no DDL)
            store = self._new_store(ds, perform_setup=False)
            self._cache[ds.id] = store
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(ds.id)
        return store

    async def create_table(self, ds: Dataset) -> None:
        store = self._new_store(ds, perform_setup=True)
        await asyncio.to_thread(store._initialize)  # creates schema/extension/table/HNSW index

    async def drop_table(self, ds: Dataset) -> None:
        self._cache.pop(ds.id, None)
        table = f"data_{vector_table_name(ds.id)}"
        async with self._db.async_engine.begin() as conn:
            await conn.execute(sa.text(f'DROP TABLE IF EXISTS {VECTOR_SCHEMA}."{table}"'))

    async def add(self, ds: Dataset, nodes: list[BaseNode]) -> None:
        if nodes:
            await self._store(ds).async_add(nodes)

    async def delete_document(self, ds: Dataset, document_id: str) -> None:
        t = _physical_table(ds)
        async with self._db.async_engine.begin() as conn:
            await conn.execute(
                sa.delete(t).where(t.c.metadata_["ref_doc_id"].astext == document_id)
            )

    async def delete_document_except_job(self, ds: Dataset, document_id: str, job_id: str) -> None:
        """Re-indexing: keep only the nodes inserted by this job and delete older nodes of the same document."""
        t = _physical_table(ds)
        job = t.c.metadata_["job_id"].astext
        async with self._db.async_engine.begin() as conn:
            await conn.execute(
                sa.delete(t).where(
                    t.c.metadata_["ref_doc_id"].astext == document_id,
                    sa.or_(job.is_(None), job != job_id),
                )
            )

    async def delete_job_nodes(self, ds: Dataset, document_id: str, job_id: str) -> None:
        """Remove only the nodes inserted by a failed/canceled job."""
        t = _physical_table(ds)
        async with self._db.async_engine.begin() as conn:
            await conn.execute(
                sa.delete(t).where(
                    t.c.metadata_["ref_doc_id"].astext == document_id,
                    t.c.metadata_["job_id"].astext == job_id,
                )
            )

    async def replace_document_metadata(
        self, ds: Dataset, document_id: str, old_keys: list[str], new_meta: dict[str, Any]
    ) -> None:
        """Apply a document metadata change to existing nodes: (metadata_ - old_keys) || new_meta."""
        t = _physical_table(ds)
        expr = t.c.metadata_
        if old_keys:
            expr = expr.op("-")(sa.cast(sa.literal(old_keys), ARRAY(sa.Text)))
        expr = expr.op("||")(sa.bindparam("new_meta", value=new_meta, type_=JSONB))
        async with self._db.async_engine.begin() as conn:
            await conn.execute(
                sa.update(t)
                .where(t.c.metadata_["ref_doc_id"].astext == document_id)
                .values(metadata_=expr)
            )

    async def count_document_nodes(self, ds: Dataset, document_id: str) -> int:
        t = _physical_table(ds)
        async with self._db.async_engine.connect() as conn:
            res = await conn.execute(
                sa.select(sa.func.count()).select_from(t).where(
                    t.c.metadata_["ref_doc_id"].astext == document_id
                )
            )
            return int(res.scalar_one())

    async def query(
        self,
        ds: Dataset,
        embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStoreQueryResult:
        query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=top_k)
        return await self._store(ds).aquery(
            query, rag_document_ids=document_ids, rag_metadata=metadata
        )
