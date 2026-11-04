"""Database engines and sessions.

Metadata uses the async engine. LlamaIndex PGVectorStore requires both a sync and an async
engine, so both are created from the same DATABASE_URL and shared.
"""

from collections.abc import AsyncIterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession


class Database:
    def __init__(self, url: str, pool_size: int = 10) -> None:
        self.url = url
        self.async_engine: AsyncEngine = create_async_engine(
            url, pool_size=pool_size, pool_pre_ping=True
        )
        self.sync_engine: Engine = create_engine(url, pool_size=2, pool_pre_ping=True)
        self.sessionmaker = async_sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessionmaker() as session:
            yield session

    async def dispose(self) -> None:
        await self.async_engine.dispose()
        self.sync_engine.dispose()
