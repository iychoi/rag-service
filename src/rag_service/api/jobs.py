"""Indexing / job API."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import func
from sqlmodel import col, select

from rag_service.api.deps import DatasetDep, DocumentDep, JobDep, ModelAPIKey, Session, State
from rag_service.errors import APIError
from rag_service.api.docs import errors
from rag_service.models.schemas import (
    BulkIndexOut,
    BulkIndexRequest,
    IndexRequest,
    JobOut,
    Page,
    ParserHealthOut,
)
from rag_service.models.tables import (
    ACTIVE_JOB_STATUSES,
    Dataset,
    Document,
    DocumentStatus,
    IndexJob,
    JobStatus,
)
from rag_service.services import jobs as job_svc

router = APIRouter(tags=["indexing"])


@router.get(
    "/parsers",
    response_model=list[ParserHealthOut],
    summary="List parsers",
    description="Checks each parser service and reports whether it is reachable.",
)
async def list_parsers(state: State) -> list[ParserHealthOut]:
    parsers = state.parsers.all()
    healths = await asyncio.gather(*(p.health() for p in parsers))
    return [
        ParserHealthOut(
            name=p.name,
            available=h.available,
            detail=h.detail,
            version=h.version,
            supported_extensions=sorted(p.supported_extensions),
        )
        for p, h in zip(parsers, healths, strict=True)
    ]


@router.post(
    "/datasets/{dataset_id}/documents/{document_id}/index",
    response_model=JobOut,
    status_code=202,
    summary="Index a document",
    description="Starts a background job (parse → chunk → embed → store). Re-indexing replaces the "
    "document's chunks without a gap; if the job fails, the previous chunks stay. Parse results are "
    "cached per parser and options unless `force_reparse` is true.",
    responses=errors(
        {
            400: ["missing_model_api_key"],
            404: ["dataset_not_found", "document_not_found"],
            409: ["job_in_progress"],
            422: ["validation_error", "unsupported_parser_or_format", "invalid_parser_options"],
            503: ["parser_unavailable", "litellm_not_configured"],
        }
    ),
)
async def index_document(
    body: IndexRequest,
    ds: DatasetDep,
    doc: DocumentDep,
    state: State,
    session: Session,
    api_key: ModelAPIKey,
) -> JobOut:
    job_svc.require_api_key_if_needed(state, ds, api_key)
    parser, options = job_svc.resolve_parser(state, body.parser, doc.filename, body.options)
    await job_svc.ensure_parser_available(parser)
    job = await job_svc.create_job(session, ds, doc, parser, options, body.force_reparse)
    state.runner.submit(job.id, api_key)
    return JobOut.from_row(job)


@router.post(
    "/datasets/{dataset_id}/index",
    response_model=BulkIndexOut,
    status_code=202,
    summary="Index documents in bulk",
    description="Creates one job per eligible document. Ineligible documents are listed in `skipped`. "
    "The whole request fails if the options are invalid or the parser is down.",
    responses=errors(
        {
            400: ["missing_model_api_key"],
            404: ["dataset_not_found"],
            422: ["validation_error", "invalid_parser_options"],
            503: ["parser_unavailable", "litellm_not_configured"],
        }
    ),
)
async def index_dataset(
    body: BulkIndexRequest,
    ds: DatasetDep,
    state: State,
    session: Session,
    api_key: ModelAPIKey,
) -> BulkIndexOut:
    """Bulk-index documents in a dataset. Documents that are not eligible are listed in `skipped` with a reason."""
    job_svc.require_api_key_if_needed(state, ds, api_key)
    ds_id = ds.id
    stmt = select(Document).where(Document.dataset_id == ds_id)
    if body.scope == "unindexed":
        stmt = stmt.where(col(Document.status).in_([DocumentStatus.uploaded, DocumentStatus.failed]))
    # A rollback expires session objects, so capture ids/filenames up front
    targets = [(d.id, d.filename) for d in (await session.exec(stmt.order_by(col(Document.created_at)))).all()]

    active_docs = set(
        (
            await session.exec(
                select(IndexJob.document_id).where(
                    IndexJob.dataset_id == ds_id, col(IndexJob.status).in_(ACTIVE_JOB_STATUSES)
                )
            )
        ).all()
    )
    checked_available: set[str] = set()
    jobs: list[JobOut] = []
    skipped: list[dict[str, str]] = []
    for doc_id, filename in targets:
        if doc_id in active_docs:
            skipped.append({"document_id": doc_id, "reason": "job_in_progress"})
            continue
        try:
            parser, options = job_svc.resolve_parser(state, body.parser, filename, body.options)
            if parser.name not in checked_available:
                await job_svc.ensure_parser_available(parser)
                checked_available.add(parser.name)
            doc = await session.get(Document, doc_id)
            if doc is None:  # deleted in the meantime
                continue
            ds = await session.get(Dataset, ds_id) or ds
            job = await job_svc.create_job(session, ds, doc, parser, options, body.force_reparse)
        except APIError as e:
            if e.code == "invalid_parser_options" or e.status_code == 503:
                raise  # the whole request is invalid or the parser is down
            skipped.append({"document_id": doc_id, "reason": e.code})
            continue
        state.runner.submit(job.id, api_key)
        jobs.append(JobOut.from_row(job))
    return BulkIndexOut(jobs=jobs, skipped=skipped)


@router.get(
    "/datasets/{dataset_id}/jobs",
    response_model=Page[JobOut],
    summary="List jobs",
    description="Filter by `document_id` to get a document's indexing history.",
    responses=errors({404: ["dataset_not_found"]}),
)
async def list_jobs(
    ds: DatasetDep,
    session: Session,
    status: Annotated[JobStatus | None, Query(description="Filter by status")] = None,
    document_id: Annotated[str | None, Query(description="Only jobs of this document")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[JobOut]:
    stmt = select(IndexJob).where(IndexJob.dataset_id == ds.id)
    if status is not None:
        stmt = stmt.where(IndexJob.status == status)
    if document_id is not None:
        stmt = stmt.where(IndexJob.document_id == document_id)
    total = (await session.exec(select(func.count()).select_from(stmt.subquery()))).one()
    rows = (
        await session.exec(stmt.order_by(col(IndexJob.created_at).desc()).limit(limit).offset(offset))
    ).all()
    return Page(items=[JobOut.from_row(j) for j in rows], total=total, limit=limit, offset=offset)


@router.get(
    "/datasets/{dataset_id}/jobs/{job_id}",
    response_model=JobOut,
    summary="Get a job",
    description="Poll this to follow status, stage, and progress.",
    responses=errors({404: ["dataset_not_found", "job_not_found"]}),
)
async def get_job(job: JobDep) -> JobOut:
    return JobOut.from_row(job)


@router.post(
    "/datasets/{dataset_id}/jobs/{job_id}/cancel",
    response_model=JobOut,
    summary="Cancel a job",
    description="Queued jobs are canceled immediately. Running jobs are interrupted and their partial "
    "chunks removed; the call returns once cleanup is done.",
    responses=errors({404: ["dataset_not_found", "job_not_found"], 409: ["job_not_active"]}),
)
async def cancel_job(job: JobDep, state: State, session: Session) -> JobOut:
    if job.status not in ACTIVE_JOB_STATUSES:
        raise APIError(409, "job_not_active", f"job is already {job.status}")
    await state.runner.cancel(job.id)
    await session.refresh(job)
    return JobOut.from_row(job)
