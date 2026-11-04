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

    # LiteLLM proxy (OpenAI-compatible). Required for /query; the key comes from each request.
    litellm_base_url: str | None = None
    llm_model: str = "claude-sonnet-5-5"
    # Comma-separated models a /query request may pick with `model`
    llm_allowed_models: str = "claude-sonnet-5-5,claude-haiku-5-5"
    llm_timeout_s: float = 120.0
    llm_max_output_tokens: int = 1024
    llm_context_window: int = 200_000

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def llm_allowed_model_list(self) -> list[str]:
        models = [m.strip() for m in self.llm_allowed_models.split(",") if m.strip()]
        return models if self.llm_model in models else [self.llm_model, *models]


@lru_cache
def get_settings() -> Settings:
    return Settings()
