"""Search API."""

import re

from fastapi import APIRouter
from sqlmodel import col, select

from rag_service.api.deps import DatasetDep, ModelAPIKey, Session, State
from rag_service.errors import APIError
from rag_service.api.docs import errors
from rag_service.models.schemas import SearchRequest, SearchResponse, SearchResult
from rag_service.models.tables import Document
from rag_service.providers.embeddings import build_embedding
from rag_service.services.jobs import require_api_key_if_needed

router = APIRouter(tags=["search"])

_FILTER_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_ID = re.compile(r"^[0-9a-f]{32}$")


@router.post(
    "/datasets/{dataset_id}/search",
    response_model=SearchResponse,
    summary="Search a dataset",
    description="Returns the chunks most similar to the query from all indexed documents of the "
    "dataset. Send `X-Model-API-Key` if the dataset's embedding model requires it.",
    responses=errors(
        {
            400: ["missing_model_api_key"],
            404: ["dataset_not_found"],
            409: ["embedding_model_unavailable"],
            422: ["validation_error", "top_k_too_large", "invalid_filter"],
        }
    ),
)
async def search(
    body: SearchRequest,
    ds: DatasetDep,
    state: State,
    session: Session,
    api_key: ModelAPIKey,
) -> SearchResponse:
    if body.top_k > state.settings.max_top_k:
        raise APIError(422, "top_k_too_large", f"top_k must be <= {state.settings.max_top_k}")
    filters = body.filters
    document_ids = filters.document_ids if filters else None
    metadata = filters.metadata if filters else None
    if document_ids is not None:
        bad = [d for d in document_ids if not _ID.match(d)]
        if bad:
            raise APIError(422, "invalid_filter", f"invalid document ids: {bad[:5]}")
        if not document_ids:
            return SearchResponse(query=body.query, results=[])
    for key in metadata or {}:
        if not _FILTER_KEY.match(key) or key.startswith("_"):
            raise APIError(422, "invalid_filter", f"invalid metadata filter key '{key}'")

    require_api_key_if_needed(state, ds, api_key)
    entry = state.catalog.get(ds.embedding_model_id)
    assert entry is not None  # checked by require_api_key_if_needed
    embed_model = build_embedding(entry, api_key)
    query_vec = await embed_model.aget_query_embedding(body.query)

    result = await state.vectors.query(
        ds, query_vec, body.top_k, document_ids=document_ids, metadata=metadata
    )
    nodes = result.nodes or []
    sims = result.similarities or [0.0] * len(nodes)

    # Join with current document rows (drops orphan vectors left by in-flight deletes)
    doc_ids = {n.ref_doc_id for n in nodes if n.ref_doc_id}
    docs = {
        d.id: d
        for d in (
            await session.exec(
                select(Document).where(
                    Document.dataset_id == ds.id, col(Document.id).in_(list(doc_ids))
                )
            )
        ).all()
    } if doc_ids else {}

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
    return SearchResponse(query=body.query, results=results)
