"""Shared dependencies: app state, DB session, API key header, dataset path checks."""

import re
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Depends, Header, Request
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from rag_service.errors import APIError, not_found
from rag_service.models.tables import Dataset, Document, IndexJob
from rag_service.state import AppState

MODEL_API_KEY_HEADER = "X-Model-API-Key"

# Node metadata keys owned by the service (user metadata cannot override them)
RESERVED_METADATA_KEYS = frozenset(
    {
        "document_id", "doc_id", "ref_doc_id", "filename", "parser", "page", "section_path",
        "chunk_index", "job_id", "header_path", "context", "_node_content", "_node_type",
    }
)  # fmt: skip
_METADATA_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_ID = re.compile(r"^[0-9a-f]{32}$")


def get_state(request: Request) -> AppState:
    return request.app.state.rag


async def get_session(state: Annotated[AppState, Depends(get_state)]) -> AsyncIterator[AsyncSession]:
    async with state.db.sessionmaker() as session:
        yield session


State = Annotated[AppState, Depends(get_state)]
Session = Annotated[AsyncSession, Depends(get_session)]


def model_api_key(
    x_model_api_key: Annotated[
        str | None,
        Header(
            alias=MODEL_API_KEY_HEADER,
            description="LiteLLM key for embedding/LLM calls. Required when the dataset's embedding "
            "model has requires_api_key=true. Never stored.",
        ),
    ] = None,
) -> str | None:
    """LiteLLM key. Whether it is required depends on the dataset's embedding provider and is checked by the caller."""
    return x_model_api_key or None


ModelAPIKey = Annotated[str | None, Depends(model_api_key)]


def validate_user_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    for key, value in meta.items():
        if not _METADATA_KEY.match(key):
            raise APIError(422, "invalid_metadata", f"invalid metadata key '{key}'")
        if key in RESERVED_METADATA_KEYS:
            raise APIError(422, "invalid_metadata", f"metadata key '{key}' is reserved")
        if not (value is None or isinstance(value, str | int | float | bool)):
            raise APIError(
                422, "invalid_metadata", f"metadata '{key}' must be a string, number, boolean or null"
            )
    return meta


def _check_id(kind: str, ident: str) -> None:
    if not _ID.match(ident):
        raise not_found(kind, ident)


async def get_dataset(dataset_id: str, session: Session) -> Dataset:
    _check_id("dataset", dataset_id)
    ds = await session.get(Dataset, dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    return ds


DatasetDep = Annotated[Dataset, Depends(get_dataset)]


async def get_document(document_id: str, ds: DatasetDep, session: Session) -> Document:
    """Return the document only if it belongs to the dataset in the path; a document id from another dataset gives 404."""
    _check_id("document", document_id)
    doc = (
        await session.exec(
            select(Document).where(Document.id == document_id, Document.dataset_id == ds.id)
        )
    ).first()
    if doc is None:
        raise not_found("document", document_id)
    return doc


DocumentDep = Annotated[Document, Depends(get_document)]


async def get_job(job_id: str, ds: DatasetDep, session: Session) -> IndexJob:
    _check_id("job", job_id)
    job = (
        await session.exec(select(IndexJob).where(IndexJob.id == job_id, IndexJob.dataset_id == ds.id))
    ).first()
    if job is None:
        raise not_found("job", job_id)
    return job


JobDep = Annotated[IndexJob, Depends(get_job)]
