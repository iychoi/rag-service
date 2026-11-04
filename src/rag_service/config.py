from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    db_pool_size: int = 10

    embedding_catalog_path: Path = Path("./config/embedding_models.yaml")

    docling_url: str = "http://localhost:5001"
    docling_concurrency: int = 2
    docling_timeout_s: float = 900.0
    docling_poll_interval_s: float = 2.0

    # MinerU (mineru-api container). Default tier on CPU is basic; use standard etc. on GPU nodes
    mineru_url: str = "http://localhost:8010"
    mineru_tier: str = "basic"
    mineru_concurrency: int = 1
    mineru_timeout_s: float = 1800.0
    mineru_poll_interval_s: float = 2.0

    job_workers: int = 4
    embed_batch_size: int = 64

    max_upload_mb: int = 100
    max_top_k: int = 50

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
