"""Indexing pipeline: parse → chunk → embed → store."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from sqlalchemy import update
from sqlmodel import select

from rag_service.db import Database
from rag_service.models.tables import (
    Dataset,
    Document,
    DocumentStatus,
    IndexJob,
    JobStage,
    JobStatus,
    utcnow,
)
from rag_service.parsers.base import ParsedDocument, ParserError
from rag_service.parsers.registry import ParserRegistry
from rag_service.providers.catalog import EmbeddingCatalog
from rag_service.providers.embeddings import build_embedding
from rag_service.services.chunking import Chunk, chunk_markdown
from rag_service.storage.blob import BlobStore
from rag_service.storage.vector import VectorStoreManager

log = logging.getLogger(__name__)

# Progress ranges per stage
_P_PARSE_END = 0.4
_P_CHUNK_END = 0.45
_P_EMBED_END = 0.9


class JobFailedError(Exception):
    """Job failure reason that is shown to the user."""


class IndexingPipeline:
    def __init__(
        self,
        db: Database,
        parsers: ParserRegistry,
        blob: BlobStore,
        vectors: VectorStoreManager,
        catalog: EmbeddingCatalog,
        embed_batch_size: int = 64,
    ) -> None:
        self.db = db
        self.parsers = parsers
        self.blob = blob
        self.vectors = vectors
        self.catalog = catalog
        self.embed_batch_size = embed_batch_size

    async def _update_job(self, job_id: str, **values: Any) -> None:
        async with self.db.sessionmaker() as s:
            await s.exec(update(IndexJob).where(IndexJob.id == job_id).values(**values))
            await s.commit()

    async def run(
        self,
        job_id: str,
        api_key: str | None,
        canceled_by_user: Callable[[], bool] = lambda: False,
    ) -> None:
        # ---- start: queued → running ----
        async with self.db.sessionmaker() as s:
            job = await s.get(IndexJob, job_id)
            if job is None or job.status != JobStatus.queued:
                return  # deleted or canceled
            doc = await s.get(Document, job.document_id)
            ds = await s.get(Dataset, job.dataset_id)
            if doc is None or ds is None:
                return
            job.status = JobStatus.running
            job.stage = JobStage.parsing
            job.started_at = utcnow()
            doc.status = DocumentStatus.indexing
            doc.updated_at = utcnow()
            s.add(job)
            s.add(doc)
            await s.commit()

        try:
            parsed, parser_name, backend = await self._parse(job, doc, ds)
            await self._update_job(job_id, stage=JobStage.chunking, progress=_P_PARSE_END)

            chunks = await asyncio.to_thread(
                chunk_markdown, parsed, ds.chunk_size, ds.chunk_overlap
            )
            await self._update_job(job_id, stage=JobStage.embedding, progress=_P_CHUNK_END)

            nodes = await self._embed(job, doc, ds, chunks, api_key, parser_name)
            await self._update_job(job_id, stage=JobStage.storing, progress=_P_EMBED_END)

            await self.vectors.add(ds, nodes)
            await self.vectors.delete_document_except_job(ds, doc.id, job_id)
        except asyncio.CancelledError:
            if canceled_by_user():
                await self._finish_unsuccessful(job_id, ds, doc.id, JobStatus.canceled, "canceled")
            else:  # server shutdown
                await self._finish_unsuccessful(job_id, ds, doc.id, JobStatus.failed, "interrupted")
            raise
        except JobFailedError as e:
            await self._finish_unsuccessful(job_id, ds, doc.id, JobStatus.failed, str(e))
            return
        except Exception as e:  # noqa: BLE001 - the job must end in a failed state whatever happened
            log.exception("indexing job %s failed", job_id)
            await self._finish_unsuccessful(
                job_id, ds, doc.id, JobStatus.failed, f"internal error: {type(e).__name__}"
            )
            return

        now = utcnow()
        async with self.db.sessionmaker() as s:
            await s.exec(
                update(IndexJob)
                .where(IndexJob.id == job_id)
                .values(status=JobStatus.succeeded, progress=1.0, finished_at=now)
            )
            await s.exec(
                update(Document)
                .where(Document.id == doc.id)
                .values(
                    status=DocumentStatus.indexed,
                    parser=parser_name,
                    parser_backend=backend,
                    num_chunks=len(nodes),
                    indexed_at=now,
                    updated_at=now,
                )
            )
            await s.commit()

    async def _parse(
        self, job: IndexJob, doc: Document, ds: Dataset
    ) -> tuple[ParsedDocument, str, str | None]:
        try:
            parser = self.parsers.resolve(job.parser, doc.filename)
        except ValueError as e:
            raise JobFailedError(str(e)) from e
        cache_key = self.blob.parse_cache_key(parser.name, job.options)
        if not job.force_reparse:
            cached = await asyncio.to_thread(self.blob.load_parsed, ds.id, doc.id, cache_key)
            if cached is not None:
                return cached, parser.name, cached.metadata.get("backend")

        path = self.blob.resolve(doc.storage_path)
        async with self.parsers.limit(parser.name):
            try:
                parsed = await parser.parse(path, doc.filename, job.options)
            except ParserError as e:
                raise JobFailedError(str(e)) from e
        await asyncio.to_thread(self.blob.save_parsed, ds.id, doc.id, cache_key, parsed)
        return parsed, parser.name, parsed.metadata.get("backend")

    async def _embed(
        self,
        job: IndexJob,
        doc: Document,
        ds: Dataset,
        chunks: list[Chunk],
        api_key: str | None,
        parser_name: str,
    ) -> list[TextNode]:
        entry = self.catalog.get(ds.embedding_model_id)
        if entry is None:
            raise JobFailedError(f"embedding model '{ds.embedding_model_id}' is not in the catalog")
        try:
            embed_model = build_embedding(entry, api_key)
        except ValueError as e:
            raise JobFailedError(str(e)) from e

        nodes: list[TextNode] = []
        total = max(1, len(chunks))
        for start in range(0, len(chunks), self.embed_batch_size):
            batch = chunks[start : start + self.embed_batch_size]
            vectors = await embed_model.aget_text_embedding_batch([c.embedding_text() for c in batch])
            for chunk, vec in zip(batch, vectors, strict=True):
                if len(vec) != ds.embedding_dim:
                    raise JobFailedError(
                        f"dimension_mismatch: model returned {len(vec)}, dataset expects {ds.embedding_dim}"
                    )
                nodes.append(self._node(job.id, parser_name, doc, chunk, vec))
            done = min(len(chunks), start + len(batch))
            await self._update_job(
                job.id,
                progress=_P_CHUNK_END + (_P_EMBED_END - _P_CHUNK_END) * done / total,
            )
        return nodes

    @staticmethod
    def _node(
        job_id: str, parser_name: str, doc: Document, chunk: Chunk, embedding: list[float]
    ) -> TextNode:
        metadata = {
            **doc.meta,
            "filename": doc.filename,
            "parser": parser_name,
            "page": chunk.page,
            "section_path": chunk.section_path,
            "chunk_index": chunk.chunk_index,
            "job_id": job_id,
        }
        return TextNode(
            text=chunk.text,
            metadata=metadata,
            embedding=embedding,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc.id)},
        )

    async def _finish_unsuccessful(
        self, job_id: str, ds: Dataset, document_id: str, status: JobStatus, error: str
    ) -> None:
        # Remove nodes inserted by this job, if any (the previous index is kept)
        try:
            await asyncio.shield(self._cleanup(job_id, ds, document_id, status, error))
        except asyncio.CancelledError:
            pass

    async def _cleanup(
        self, job_id: str, ds: Dataset, document_id: str, status: JobStatus, error: str
    ) -> None:
        try:
            await self.vectors.delete_job_nodes(ds, document_id, job_id)
        except Exception:  # noqa: BLE001
            log.exception("failed to clean up vectors of job %s", job_id)
        async with self.db.sessionmaker() as s:
            await s.exec(
                update(IndexJob)
                .where(IndexJob.id == job_id)
                .values(status=status, error=error, finished_at=utcnow())
            )
            doc = (await s.exec(select(Document).where(Document.id == document_id))).first()
            if doc is not None:
                # If it was indexed before, the previous index is still there, so keep it indexed
                doc.status = DocumentStatus.indexed if doc.indexed_at else DocumentStatus.failed
                doc.updated_at = utcnow()
                s.add(doc)
            await s.commit()
