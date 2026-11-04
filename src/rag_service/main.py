"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from sqlalchemy import text
from sqlmodel import select

from rag_service.api import datasets, documents, jobs, search
from rag_service.api.deps import State
from rag_service.api.docs import API_DESCRIPTION, errors, install_openapi
from rag_service.config import Settings, get_settings
from rag_service.db import Database
from rag_service.errors import APIError, install_error_handlers
from rag_service.jobs.runner import JobRunner
from rag_service.models.tables import Dataset
from rag_service.parsers.docling import DoclingParser
from rag_service.parsers.mineru import MinerUParser
from rag_service.parsers.passthrough import PassthroughParser
from rag_service.parsers.registry import ParserRegistry
from rag_service.providers.catalog import EmbeddingCatalog
from rag_service.providers.embeddings import PROVIDERS
from rag_service.services.indexing import IndexingPipeline
from rag_service.state import AppState
from rag_service.storage.blob import BlobStore
from rag_service.storage.vector import VectorStoreManager

log = logging.getLogger("rag_service")


def load_catalog(settings: Settings) -> EmbeddingCatalog:
    catalog = EmbeddingCatalog.load(settings.embedding_catalog_path)
    unknown = sorted({e.provider for e in catalog.all()} - set(PROVIDERS))
    if unknown:
        raise RuntimeError(f"embedding catalog uses unimplemented providers: {unknown}")
    return catalog


async def verify_catalog_against_datasets(db: Database, catalog: EmbeddingCatalog) -> None:
    """Fail startup if a catalog entry used by existing datasets was removed or changed."""
    async with db.sessionmaker() as s:
        rows = (
            await s.exec(
                select(
                    Dataset.embedding_model_id, Dataset.embedding_model, Dataset.embedding_dim
                ).distinct()
            )
        ).all()
    problems = []
    for model_id, model, dim in rows:
        entry = catalog.get(model_id)
        if entry is None:
            problems.append(f"'{model_id}' is used by datasets but missing from catalog")
        elif entry.model != model or entry.dim != dim:
            problems.append(
                f"'{model_id}' changed (datasets use model={model} dim={dim}, "
                f"catalog has model={entry.model} dim={entry.dim})"
            )
    if problems:
        raise RuntimeError("embedding catalog mismatch: " + "; ".join(problems))


def build_state(settings: Settings) -> AppState:
    db = Database(settings.database_url, settings.db_pool_size)
    catalog = load_catalog(settings)
    parsers = ParserRegistry()
    parsers.register(PassthroughParser(), concurrency=8)
    parsers.register(
        DoclingParser(
            settings.docling_url,
            timeout_s=settings.docling_timeout_s,
            poll_interval_s=settings.docling_poll_interval_s,
        ),
        concurrency=settings.docling_concurrency,
    )
    parsers.register(
        MinerUParser(
            settings.mineru_url,
            default_tier=settings.mineru_tier,
            timeout_s=settings.mineru_timeout_s,
            poll_interval_s=settings.mineru_poll_interval_s,
        ),
        concurrency=settings.mineru_concurrency,
    )
    blob = BlobStore(settings.data_dir)
    vectors = VectorStoreManager(db)
    pipeline = IndexingPipeline(
        db, parsers, blob, vectors, catalog, embed_batch_size=settings.embed_batch_size
    )
    runner = JobRunner(db, pipeline, vectors, workers=settings.job_workers)
    return AppState(
        settings=settings,
        db=db,
        catalog=catalog,
        parsers=parsers,
        blob=blob,
        vectors=vectors,
        runner=runner,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = build_state(settings)
        app.state.rag = state
        await verify_catalog_against_datasets(state.db, state.catalog)
        await state.runner.start()
        try:
            yield
        finally:
            await state.runner.stop()
            for parser in state.parsers.all():
                if hasattr(parser, "aclose"):
                    await parser.aclose()
            await state.db.dispose()

    app = FastAPI(
        title="RAG Service",
        version="0.1.0",
        description=API_DESCRIPTION,
        lifespan=lifespan,
        generate_unique_id_function=lambda route: route.name,  # operationId = handler name
    )
    install_error_handlers(app)
    install_openapi(app)

    v1 = APIRouter(prefix="/v1")
    v1.include_router(datasets.router)
    v1.include_router(documents.router)
    v1.include_router(jobs.router)
    v1.include_router(search.router)
    app.include_router(v1)

    @app.get("/healthz", tags=["ops"], summary="Liveness probe")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/readyz",
        tags=["ops"],
        summary="Readiness probe",
        description="Checks database connectivity. Parser availability is reported by /v1/parsers.",
        responses=errors({503: ["db_unavailable"]}),
    )
    async def readyz(state: State) -> dict[str, str]:
        try:
            async with state.db.async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as e:
            raise APIError(503, "db_unavailable", "database is not reachable") from e
        return {"status": "ready"}

    return app

