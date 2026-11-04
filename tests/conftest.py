import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from testcontainers.community.postgres import PostgresContainer

from rag_service.config import Settings
from rag_service.main import create_app
from rag_service.state import AppState

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
DOCLING_TEST_URL = "http://docling.test"
MINERU_TEST_URL = "http://mineru.test"
LITELLM_TEST_URL = "http://litellm.test/v1"


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    with PostgresContainer("pgvector/pgvector:pg17", driver="psycopg") as pg:
        url = pg.get_connection_url()
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.attributes["database_url"] = url
        command.upgrade(cfg, "head")
        yield url


@pytest.fixture(scope="session")
def settings(database_url: str, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        database_url=database_url,
        data_dir=tmp_path_factory.mktemp("data"),
        docling_url=DOCLING_TEST_URL,
        docling_poll_interval_s=0.01,
        mineru_url=MINERU_TEST_URL,
        mineru_poll_interval_s=0.01,
        litellm_base_url=LITELLM_TEST_URL,
        litellm_max_retries=1,
        job_workers=2,
    )


@pytest.fixture(scope="session")
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture(scope="session")
def state(app: FastAPI) -> AppState:
    return app.state.rag


@pytest.fixture(scope="session")
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/v1") as c:
        yield c


@pytest.fixture
async def dataset(client: httpx.AsyncClient) -> AsyncIterator[dict]:
    r = await client.post(
        "/datasets", json={"name": "test", "embedding": {"provider": "hash"}, "chunk_size": 256}
    )
    assert r.status_code == 201, r.text
    ds = r.json()
    yield ds
    await client.delete(f"/datasets/{ds['id']}")


async def wait_for_job(client: httpx.AsyncClient, ds_id: str, job_id: str, timeout: float = 30) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        r = await client.get(f"/datasets/{ds_id}/jobs/{job_id}")
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] not in ("queued", "running"):
            return job
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"job {job_id} still {job['status']}")
        await asyncio.sleep(0.05)
