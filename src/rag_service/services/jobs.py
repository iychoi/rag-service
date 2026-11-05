"""Rules for creating indexing jobs."""

from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlmodel.ext.asyncio.session import AsyncSession

from rag_service.errors import APIError
from rag_service.models.tables import Dataset, Document, IndexJob
from rag_service.parsers.base import DocumentParser
from rag_service.services.embedding import require_embedding_access
from rag_service.state import AppState


def require_api_key_if_needed(state: AppState, ds: Dataset, api_key: str | None) -> None:
    require_embedding_access(state, ds.embedding_provider, api_key)
    if ds.contextual_model:  # contexts are written through LiteLLM even with hash embeddings
        require_embedding_access(state, "litellm", api_key)


def resolve_parser(
    state: AppState, parser: str, filename: str, options: dict[str, Any]
) -> tuple[DocumentParser, dict[str, Any]]:
    try:
        resolved = state.parsers.resolve(parser, filename)
    except ValueError as e:
        raise APIError(422, "unsupported_parser_or_format", str(e)) from e
    if resolved.name == "passthrough":
        return resolved, {}  # md/txt files ignore parser options
    try:
        return resolved, resolved.validate_options(options)
    except ValueError as e:
        raise APIError(422, "invalid_parser_options", str(e)) from e


async def ensure_parser_available(parser: DocumentParser) -> None:
    health = await parser.health()
    if not health.available:
        raise APIError(503, "parser_unavailable", f"parser '{parser.name}' is unavailable")


async def create_job(
    session: AsyncSession,
    ds: Dataset,
    doc: Document,
    parser: DocumentParser,
    options: dict[str, Any],
    force_reparse: bool,
) -> IndexJob:
    """Create and commit the job row. Returns 409 if the document already has an active job.

    Duplicates are prevented by the partial unique index uq_index_jobs_active_document.
    """
    doc_id = doc.id  # capture before a possible rollback expires `doc`
    job = IndexJob(
        dataset_id=ds.id,
        document_id=doc_id,
        parser=parser.name,
        options=options,
        force_reparse=force_reparse,
    )
    doc.last_job_id = job.id
    session.add(job)
    session.add(doc)
    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        raise APIError(
            409, "job_in_progress", f"document '{doc_id}' already has an active indexing job"
        ) from e
    await session.refresh(job)
    return job
