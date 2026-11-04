from dataclasses import dataclass

from rag_service.config import Settings
from rag_service.db import Database
from rag_service.jobs.runner import JobRunner
from rag_service.parsers.registry import ParserRegistry
from rag_service.providers.catalog import EmbeddingCatalog
from rag_service.storage.blob import BlobStore
from rag_service.storage.vector import VectorStoreManager


@dataclass
class AppState:
    settings: Settings
    db: Database
    catalog: EmbeddingCatalog
    parsers: ParserRegistry
    blob: BlobStore
    vectors: VectorStoreManager
    runner: JobRunner
