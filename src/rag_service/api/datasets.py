"""Dataset API."""

from typing import Annotated

from fastapi import APIRouter, Query, Response
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import col, select

from rag_service.api.deps import DatasetDep, Session, State, validate_user_metadata
from rag_service.api.docs import errors
from rag_service.errors import APIError
from rag_service.models.schemas import (
    DatasetCreate,
    DatasetOut,
    DatasetUpdate,
    EmbeddingModelOut,
    Page,
)
from rag_service.models.tables import ACTIVE_JOB_STATUSES, Dataset, Document, IndexJob, new_id, utcnow
from rag_service.providers.embeddings import provider_requires_api_key
from rag_service.storage.vector import vector_table_name

router = APIRouter(tags=["datasets"])


@router.get(
    "/embedding-models",
    response_model=list[EmbeddingModelOut],
    summary="List embedding models",
    description="Embedding models a dataset can be created with. `requires_api_key` tells whether "
    "indexing and search requests must send `X-Model-API-Key`.",
)
async def list_embedding_models(state: State) -> list[EmbeddingModelOut]:
    return [
        EmbeddingModelOut(
            id=e.id,
            provider=e.provider,
            model=e.model,
            dim=e.dim,
            max_input_tokens=e.max_input_tokens,
            requires_api_key=provider_requires_api_key(e.provider),
            description=e.description,
        )
        for e in state.catalog.all()
    ]


async def _document_counts(session: Session, dataset_id: str) -> dict[str, int]:
    rows = (
        await session.exec(
            select(Document.status, func.count())
            .where(Document.dataset_id == dataset_id)
            .group_by(Document.status)
        )
    ).all()
    counts = {str(status): int(n) for status, n in rows}
    counts["total"] = sum(counts.values())
    return counts


@router.post(
    "/datasets",
    response_model=DatasetOut,
    status_code=201,
    summary="Create a dataset",
    description="The embedding model is chosen here and cannot be changed later. "
    "The dataset's vector table is created immediately.",
    responses=errors(
        {
            409: ["external_ref_conflict"],
            422: ["validation_error", "unknown_embedding_model", "chunk_size_too_large",
                  "invalid_chunk_overlap", "invalid_metadata"],
        }
    ),  # fmt: skip
)
async def create_dataset(body: DatasetCreate, state: State, session: Session) -> DatasetOut:
    entry = state.catalog.get(body.embedding_model_id)
    if entry is None:
        raise APIError(
            422,
            "unknown_embedding_model",
            f"unknown embedding_model_id '{body.embedding_model_id}' "
            f"(available: {[e.id for e in state.catalog.all()]})",
        )
    if body.chunk_size > entry.max_input_tokens:
        raise APIError(
            422,
            "chunk_size_too_large",
            f"chunk_size {body.chunk_size} exceeds model max input ({entry.max_input_tokens} tokens)",
        )
    if body.chunk_overlap >= body.chunk_size:
        raise APIError(422, "invalid_chunk_overlap", "chunk_overlap must be smaller than chunk_size")
    validate_user_metadata(body.metadata)

    ds_id = new_id()
    ds = Dataset(
        id=ds_id,
        name=body.name,
        external_ref=body.external_ref,
        description=body.description,
        embedding_model_id=entry.id,
        embedding_provider=entry.provider,
        embedding_model=entry.model,
        embedding_dim=entry.dim,
        vector_table=vector_table_name(ds_id),
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        meta=body.metadata,
    )
    session.add(ds)
    try:
        await session.flush()
    except IntegrityError as e:
        await session.rollback()
        raise APIError(
            409, "external_ref_conflict", f"external_ref '{body.external_ref}' already exists"
        ) from e

    try:
        await state.vectors.create_table(ds)
    except Exception:
        await session.rollback()
        raise
    try:
        await session.commit()
    except Exception:
        await state.vectors.drop_table(ds)
        raise
    return DatasetOut.from_row(ds, {"total": 0})


@router.get(
    "/datasets",
    response_model=Page[DatasetOut],
    summary="List datasets",
    description="Newest first. A gateway can pass the ids a caller may see in `ids`.",
)
async def list_datasets(
    session: Session,
    ids: Annotated[list[str] | None, Query(description="Filter by dataset id (repeat or comma-separate)")] = None,
    external_ref: Annotated[str | None, Query(description="Filter by external reference")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[DatasetOut]:
    stmt = select(Dataset)
    if ids is not None:
        flat = [i for item in ids for i in item.split(",") if i]
        stmt = stmt.where(col(Dataset.id).in_(flat))
    if external_ref is not None:
        stmt = stmt.where(Dataset.external_ref == external_ref)
    total = (await session.exec(select(func.count()).select_from(stmt.subquery()))).one()
    rows = (
        await session.exec(stmt.order_by(col(Dataset.created_at).desc()).limit(limit).offset(offset))
    ).all()
    return Page(items=[DatasetOut.from_row(r) for r in rows], total=total, limit=limit, offset=offset)


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetOut,
    summary="Get a dataset",
    description="Includes document counts per status.",
    responses=errors({404: ["dataset_not_found"]}),
)
async def get_dataset(ds: DatasetDep, session: Session) -> DatasetOut:
    return DatasetOut.from_row(ds, await _document_counts(session, ds.id))


@router.patch(
    "/datasets/{dataset_id}",
    response_model=DatasetOut,
    summary="Update a dataset",
    description="Only name, description, and metadata can change; metadata replaces the whole object.",
    responses=errors({404: ["dataset_not_found"], 422: ["validation_error", "invalid_metadata"]}),
)
async def update_dataset(body: DatasetUpdate, ds: DatasetDep, session: Session) -> DatasetOut:
    if body.name is not None:
        ds.name = body.name
    if "description" in body.model_fields_set:
        ds.description = body.description
    if body.metadata is not None:
        ds.meta = validate_user_metadata(body.metadata)
    ds.updated_at = utcnow()
    session.add(ds)
    await session.commit()
    await session.refresh(ds)
    return DatasetOut.from_row(ds, await _document_counts(session, ds.id))


@router.delete(
    "/datasets/{dataset_id}",
    status_code=204,
    summary="Delete a dataset",
    description="Cancels active jobs, then deletes the vector table, documents, jobs, stored files, "
    "and parse caches. Cannot be undone.",
    responses=errors({404: ["dataset_not_found"]}),
)
async def delete_dataset(ds: DatasetDep, state: State, session: Session) -> Response:
    active = (
        await session.exec(
            select(IndexJob.id).where(
                IndexJob.dataset_id == ds.id, col(IndexJob.status).in_(ACTIVE_JOB_STATUSES)
            )
        )
    ).all()
    await state.runner.cancel_many(list(active))

    await state.vectors.drop_table(ds)
    await session.delete(ds)  # documents and index_jobs are removed via FK CASCADE
    await session.commit()
    state.blob.delete_dataset(ds.id)
    return Response(status_code=204)
