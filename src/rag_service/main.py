"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from sqlalchemy import text

from rag_service.api import datasets, documents, jobs, search
from rag_service.api.deps import State
from rag_service.api.docs import API_DESCRIPTION, errors, install_openapi
from rag_service.config import Settings, get_settings
from rag_service.db import Database
from rag_service.errors import APIError, install_error_handlers
from rag_service.jobs.runner import JobRunner
from rag_service.parsers.docling import DoclingParser
from rag_service.parsers.mineru import MinerUParser
from rag_service.parsers.passthrough import PassthroughParser
from rag_service.parsers.registry import ParserRegistry
from rag_service.services.indexing import IndexingPipeline
from rag_service.state import AppState
from rag_service.storage.blob import BlobStore
from rag_service.storage.vector import VectorStoreManager

log = logging.getLogger("rag_service")


def build_state(settings: Settings) -> AppState:
    db = Database(settings.database_url, settings.db_pool_size)
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
    pipeline = IndexingPipeline(db, parsers, blob, vectors, settings)
    runner = JobRunner(db, pipeline, vectors, workers=settings.job_workers)
    return AppState(
        settings=settings,
        db=db,
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

