"""Retrieval shared by /search and /query: validate filters, embed the query, search, join documents."""

import re

from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from rag_service.errors import APIError
from rag_service.models.schemas import SearchFilters, SearchResult
from rag_service.models.tables import Dataset, Document
from rag_service.services.embedding import embed_query
from rag_service.state import AppState

_FILTER_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_ID = re.compile(r"^[0-9a-f]{32}$")


def validate_retrieval_request(state: AppState, top_k: int, filters: SearchFilters | None) -> None:
    if top_k > state.settings.max_top_k:
        raise APIError(422, "top_k_too_large", f"top_k must be <= {state.settings.max_top_k}")
    if filters is None:
        return
    if filters.document_ids:
        bad = [d for d in filters.document_ids if not _ID.match(d)]
        if bad:
            raise APIError(422, "invalid_filter", f"invalid document ids: {bad[:5]}")
    for key in filters.metadata or {}:
        if not _FILTER_KEY.match(key) or key.startswith("_"):
            raise APIError(422, "invalid_filter", f"invalid metadata filter key '{key}'")


async def retrieve(
    state: AppState,
    session: AsyncSession,
    ds: Dataset,
    query: str,
    top_k: int,
    filters: SearchFilters | None,
    api_key: str | None,
) -> list[SearchResult]:
    """Return the chunks most similar to `query`, best first. Call validate_retrieval_request first."""
    document_ids = filters.document_ids if filters else None
    metadata = filters.metadata if filters else None
    if document_ids is not None and not document_ids:
        return []

    query_vec = await embed_query(state, ds, query, api_key)

    result = await state.vectors.query(
        ds, query_vec, top_k, document_ids=document_ids, metadata=metadata
    )
    nodes = result.nodes or []
    sims = result.similarities or [0.0] * len(nodes)

    # Join with current document rows (drops orphan vectors left by in-flight deletes)
    doc_ids = {n.ref_doc_id for n in nodes if n.ref_doc_id}
    docs: dict[str, Document] = {}
    if doc_ids:
        rows = await session.exec(
            select(Document).where(Document.dataset_id == ds.id, col(Document.id).in_(list(doc_ids)))
        )
        docs = {d.id: d for d in rows.all()}

    results: list[SearchResult] = []
    for node, score in zip(nodes, sims, strict=True):
        doc = docs.get(node.ref_doc_id or "")
        if doc is None:
            continue
        md = node.metadata
        results.append(
            SearchResult(
                score=float(score),
                text=node.get_content(),
                document_id=doc.id,
                filename=doc.filename,
                page=md.get("page"),
                section_path=list(md.get("section_path") or []),
                chunk_index=md.get("chunk_index"),
                metadata=doc.meta,
            )
        )
    return results
