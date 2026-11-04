"""Document (file) API."""

import json
import mimetypes
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import col, select

from rag_service.api.deps import (
    DatasetDep,
    DocumentDep,
    ModelAPIKey,
    Session,
    State,
    validate_user_metadata,
)
from rag_service.errors import APIError
from rag_service.api.docs import errors
from rag_service.models.schemas import (
    DocumentOut,
    DocumentUpdate,
    DocumentUploadOut,
    JobOut,
    Page,
)
from rag_service.models.tables import (
    ACTIVE_JOB_STATUSES,
    Document,
    DocumentStatus,
    IndexJob,
    new_id,
    utcnow,
)
from rag_service.services import jobs as job_svc
from rag_service.storage.blob import UploadTooLargeError

router = APIRouter(prefix="/datasets/{dataset_id}/documents", tags=["documents"])


def _parse_json_field(name: str, raw: str | None) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise APIError(422, "invalid_json", f"form field '{name}' is not valid JSON") from e
    if not isinstance(value, dict):
        raise APIError(422, "invalid_json", f"form field '{name}' must be a JSON object")
    return value


def _safe_filename(name: str | None) -> str:
    base = Path(name or "").name.strip()
    if not base or base in {".", ".."}:
        raise APIError(422, "invalid_filename", "uploaded file must have a filename")
    return base[:1024]


@router.post(
    "",
    response_model=DocumentUploadOut,
    status_code=201,
    summary="Upload a document",
    description="Multipart upload. With `index=true` an indexing job starts right away; the indexing "
    "request is validated before the file is stored. `.md`/`.markdown`/`.txt` always use the "
    "passthrough parser. Send `X-Model-API-Key` with `index=true` if the dataset's model needs it.",
    responses=errors(
        {
            400: ["missing_model_api_key"],
            404: ["dataset_not_found"],
            409: ["duplicate_document"],
            413: ["file_too_large"],
            422: ["validation_error", "invalid_filename", "invalid_json", "invalid_metadata",
                  "unsupported_parser_or_format", "invalid_parser_options"],
            503: ["parser_unavailable", "litellm_not_configured"],
        }
    ),  # fmt: skip
)
async def upload_document(
    ds: DatasetDep,
    state: State,
    session: Session,
    api_key: ModelAPIKey,
    file: Annotated[UploadFile, File(description="The document (max MAX_UPLOAD_MB, default 100 MB)")],
    metadata: Annotated[
        str | None, Form(description='Document metadata as a JSON object, e.g. {"year": 2025}')
    ] = None,
    index: Annotated[bool, Form(description="Start an indexing job right after the upload")] = False,
    parser: Annotated[
        str, Form(description="docling or mineru; used when index=true (md/txt use passthrough)")
    ] = "docling",
    options: Annotated[
        str | None,
        Form(description='Parser options as a JSON object, e.g. {"do_ocr": false}; used when index=true'),
    ] = None,
) -> DocumentUploadOut:
    filename = _safe_filename(file.filename)
    meta = validate_user_metadata(_parse_json_field("metadata", metadata))
    parse_options = _parse_json_field("options", options)

    # If indexing was requested too, validate the request before storing the file (fail fast)
    resolved = None
    if index:
        job_svc.require_api_key_if_needed(state, ds, api_key)
        resolved, parse_options = job_svc.resolve_parser(state, parser, filename, parse_options)
        await job_svc.ensure_parser_available(resolved)

    try:
        stored = await state.blob.receive_upload(file, state.settings.max_upload_bytes)
    except UploadTooLargeError as e:
        raise APIError(
            413, "file_too_large", f"file exceeds {state.settings.max_upload_mb} MB limit"
        ) from e

    existing = (
        await session.exec(
            select(Document.id).where(Document.dataset_id == ds.id, Document.sha256 == stored.sha256)
        )
    ).first()
    if existing:
        state.blob.discard_upload(stored)
        raise APIError(
            409, "duplicate_document", f"identical file already exists as document '{existing}'"
        )

    doc_id = new_id()
    dest = state.blob.original_path(ds.id, doc_id, filename)
    doc = Document(
        id=doc_id,
        dataset_id=ds.id,
        filename=filename,
        mime_type=file.content_type
        or mimetypes.guess_type(filename)[0]
        or "application/octet-stream",
        size_bytes=stored.size,
        sha256=stored.sha256,
        storage_path=state.blob.relative(dest),
        meta=meta,
    )
    session.add(doc)
    try:
        await session.flush()
    except IntegrityError as e:  # concurrent upload race
        await session.rollback()
        state.blob.discard_upload(stored)
        raise APIError(409, "duplicate_document", "identical file already exists") from e
    state.blob.commit_upload(stored, dest)
    try:
        await session.commit()
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    await session.refresh(doc)

    job = None
    if index and resolved is not None:
        job = await job_svc.create_job(session, ds, doc, resolved, parse_options, False)
        state.runner.submit(job.id, api_key)
        await session.refresh(doc)
    return DocumentUploadOut(
        document=DocumentOut.from_row(doc, job), job=JobOut.from_row(job) if job else None
    )


@router.get(
    "",
    response_model=Page[DocumentOut],
    summary="List documents",
    responses=errors({404: ["dataset_not_found"]}),
)
async def list_documents(
    ds: DatasetDep,
    session: Session,
    status: Annotated[DocumentStatus | None, Query(description="Filter by status")] = None,
    parser: Annotated[
        str | None, Query(description="Filter by parser (docling, mineru, passthrough)")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[DocumentOut]:
    stmt = select(Document).where(Document.dataset_id == ds.id)
    if status is not None:
        stmt = stmt.where(Document.status == status)
    if parser is not None:
        stmt = stmt.where(Document.parser == parser)
    total = (await session.exec(select(func.count()).select_from(stmt.subquery()))).one()
    rows = (
        await session.exec(
            stmt.order_by(col(Document.created_at).desc()).limit(limit).offset(offset)
        )
    ).all()
    return Page(
        items=[DocumentOut.from_row(d) for d in rows], total=total, limit=limit, offset=offset
    )


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Get a document",
    description="Includes the most recent indexing job in `last_job`.",
    responses=errors({404: ["dataset_not_found", "document_not_found"]}),
)
async def get_document(doc: DocumentDep, session: Session) -> DocumentOut:
    job = await session.get(IndexJob, doc.last_job_id) if doc.last_job_id else None
    return DocumentOut.from_row(doc, job)


@router.get(
    "/{document_id}/content",
    summary="Download the original file",
    description="Returns the file as uploaded with its original content type.",
    response_class=FileResponse,
    responses={
        200: {
            "description": "The original file",
            "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
        },
        **errors({404: ["dataset_not_found", "document_not_found", "content_missing"]}),
    },
)
async def download_document(doc: DocumentDep, state: State) -> FileResponse:
    path = state.blob.resolve(doc.storage_path)
    if not path.exists():
        raise APIError(404, "content_missing", "stored file is missing")
    return FileResponse(path, media_type=doc.mime_type, filename=doc.filename)


@router.get(
    "/{document_id}/parsed",
    response_class=PlainTextResponse,
    summary="Get the parsed markdown",
    description="Markdown produced by the parser for the most recent successful indexing job.",
    responses={
        200: {
            "description": "Parsed markdown",
            "content": {
                "text/markdown": {
                    "schema": {"type": "string"},
                    "example": "## Article 5. Termination\n\nEither party may terminate with 30 days written notice.",
                }
            },
        },
        **errors({404: ["dataset_not_found", "document_not_found", "not_parsed"]}),
    },
)
async def get_parsed(doc: DocumentDep, ds: DatasetDep, state: State, session: Session) -> PlainTextResponse:
    """Parsed markdown used by the most recent successful indexing job."""
    job = (
        await session.exec(
            select(IndexJob)
            .where(IndexJob.document_id == doc.id, IndexJob.status == "succeeded")
            .order_by(col(IndexJob.finished_at).desc())
        )
    ).first()
    if job is None:
        raise APIError(404, "not_parsed", "document has not been parsed yet")
    key = state.blob.parse_cache_key(job.parser, job.options)
    md = state.blob.latest_parsed_markdown(ds.id, doc.id, key)
    if md is None:
        raise APIError(404, "not_parsed", "parsed content is not available")
    return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")


@router.patch(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Update document metadata",
    description="Replaces the metadata. Indexed chunks are updated immediately, so search filters "
    "see the change without re-indexing.",
    responses=errors(
        {404: ["dataset_not_found", "document_not_found"], 422: ["validation_error", "invalid_metadata"]}
    ),
)
async def update_document(
    body: DocumentUpdate, doc: DocumentDep, ds: DatasetDep, state: State, session: Session
) -> DocumentOut:
    new_meta = validate_user_metadata(body.metadata)
    old_meta = doc.meta
    doc.meta = new_meta
    doc.updated_at = utcnow()
    session.add(doc)
    await session.commit()
    # Update vector metadata as well so search filters reflect the change immediately
    await state.vectors.replace_document_metadata(ds, doc.id, list(old_meta), new_meta)
    await session.refresh(doc)
    return DocumentOut.from_row(doc)


@router.delete(
    "/{document_id}",
    status_code=204,
    summary="Delete a document",
    description="Cancels an active job, then deletes the chunks, record, jobs, stored file, and parse cache.",
    responses=errors({404: ["dataset_not_found", "document_not_found"]}),
)
async def delete_document(
    doc: DocumentDep, ds: DatasetDep, state: State, session: Session
) -> Response:
    active = (
        await session.exec(
            select(IndexJob.id).where(
                IndexJob.document_id == doc.id, col(IndexJob.status).in_(ACTIVE_JOB_STATUSES)
            )
        )
    ).all()
    await state.runner.cancel_many(list(active))
    # Order: vectors → metadata → files
    await state.vectors.delete_document(ds, doc.id)
    storage_path = doc.storage_path
    await session.delete(doc)
    await session.commit()
    state.blob.delete_document(ds.id, doc.id, storage_path)
    return Response(status_code=204)
