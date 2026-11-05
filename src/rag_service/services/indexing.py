"""Indexing pipeline: parse → chunk → (contextualize) → embed → store."""

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
from rag_service.config import Settings
from rag_service.providers.embeddings import build_embedding
from rag_service.providers.upstream import upstream_api_error
from rag_service.services.embedding import spec_from_dataset
from rag_service.services.chunking import Chunk, chunk_document
from rag_service.services.contextual import ContextGenerator
from rag_service.storage.blob import BlobStore
from rag_service.storage.vector import VectorStoreManager

log = logging.getLogger(__name__)

# Progress ranges per stage. With contextual retrieval, 0.45–0.7 is contextualizing and
# embedding starts at 0.7.
_P_PARSE_END = 0.4
_P_CHUNK_END = 0.45
_P_CONTEXT_END = 0.7
_P_EMBED_END = 0.9


class JobFailedError(Exception):
    """Job failure reason that is shown to the user."""


class DocumentDeletedError(Exception):
    """The document was deleted while its job was running."""


class IndexingPipeline:
    def __init__(
        self,
        db: Database,
        parsers: ParserRegistry,
        blob: BlobStore,
        vectors: VectorStoreManager,
        settings: Settings,
    ) -> None:
        self.db = db
        self.parsers = parsers
        self.blob = blob
        self.vectors = vectors
        self.settings = settings
        self.embed_batch_size = settings.embed_batch_size

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
                chunk_document, parsed, parser_name, ds.chunk_size, ds.chunk_overlap
            )
            contexts: list[str] | None = None
            embed_start = _P_CHUNK_END
            if ds.contextual_model and chunks:
                await self._update_job(job_id, stage=JobStage.contextualizing, progress=_P_CHUNK_END)
                contexts = await self._contextualize(job_id, ds, chunks, api_key)
                embed_start = _P_CONTEXT_END
            await self._update_job(job_id, stage=JobStage.embedding, progress=embed_start)

            nodes = await self._embed(
                job, doc, ds, chunks, contexts, api_key, parser_name, embed_start
            )
            await self._update_job(job_id, stage=JobStage.storing, progress=_P_EMBED_END)

            await self.vectors.add(ds, nodes)
            await self.vectors.delete_document_except_job(ds, doc.id, job_id)
            await self._ensure_document_exists(doc.id)
        except DocumentDeletedError:
            await self._discard_output(job_id, ds, doc.id)
            return
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
        await self._ensure_document_exists(doc.id)
        return parsed, parser.name, parsed.metadata.get("backend")

    async def _contextualize(
        self, job_id: str, ds: Dataset, chunks: list[Chunk], api_key: str | None
    ) -> list[str]:
        s = self.settings
        if not s.litellm_base_url or not api_key:
            raise JobFailedError("context generation needs LITELLM_BASE_URL and X-Model-API-Key")
        assert ds.contextual_model is not None
        generator = ContextGenerator(
            base_url=s.litellm_base_url,
            model=ds.contextual_model,
            api_key=api_key,
            max_output_tokens=s.contextual_max_output_tokens,
            document_tokens=s.contextual_document_tokens,
            concurrency=s.contextual_concurrency,
            timeout_s=s.llm_timeout_s,
            max_retries=s.litellm_max_retries,
        )
        total = len(chunks)
        step = max(1, total // 20)  # about 20 progress updates per document

        async def progress(done: int) -> None:
            if done % step == 0 or done == total:
                frac = done / total
                await self._update_job(
                    job_id, progress=_P_CHUNK_END + (_P_CONTEXT_END - _P_CHUNK_END) * frac
                )

        try:
            contexts = await generator.generate([c.text for c in chunks], progress)
        except Exception as e:
            if (err := upstream_api_error(e, s.llm_timeout_s)) is not None:
                raise JobFailedError(f"context generation failed ({err.code}): {err.message}") from e
            raise
        u = generator.usage
        log.info(
            "job %s: %d contexts with %s; input %d tokens (%d cached), output %d",
            job_id, u.calls, ds.contextual_model, u.input_tokens, u.cached_input_tokens,
            u.output_tokens,
        )  # fmt: skip
        return contexts

    async def _embed(
        self,
        job: IndexJob,
        doc: Document,
        ds: Dataset,
        chunks: list[Chunk],
        contexts: list[str] | None,
        api_key: str | None,
        parser_name: str,
        start: float,
    ) -> list[TextNode]:
        s = self.settings
        try:
            embed_model = build_embedding(
                spec_from_dataset(ds),
                purpose="document",
                api_key=api_key,
                base_url=s.litellm_base_url,
                batch_size=s.embed_batch_size,
                timeout_s=s.embedding_timeout_s,
                max_retries=s.litellm_max_retries,
            )
        except ValueError as e:
            raise JobFailedError(str(e)) from e

        nodes: list[TextNode] = []
        total = max(1, len(chunks))
        for first in range(0, len(chunks), self.embed_batch_size):
            batch = chunks[first : first + self.embed_batch_size]
            batch_contexts = (contexts or [""] * len(chunks))[first : first + len(batch)]
            try:
                vectors = await embed_model.aget_text_embedding_batch(
                    [c.embedding_text(ctx) for c, ctx in zip(batch, batch_contexts, strict=True)]
                )
            except Exception as e:
                if (err := upstream_api_error(e, s.embedding_timeout_s)) is not None:
                    raise JobFailedError(f"embedding failed ({err.code}): {err.message}") from e
                raise
            for chunk, ctx, vec in zip(batch, batch_contexts, vectors, strict=True):
                if len(vec) != ds.embedding_dim:
                    raise JobFailedError(
                        f"dimension_mismatch: model returned {len(vec)}, dataset expects {ds.embedding_dim}"
                    )
                nodes.append(self._node(job.id, parser_name, doc, chunk, vec, ctx))
            done = min(len(chunks), first + len(batch))
            await self._update_job(
                job.id,
                progress=start + (_P_EMBED_END - start) * done / total,
            )
        return nodes

    @staticmethod
    def _node(
        job_id: str,
        parser_name: str,
        doc: Document,
        chunk: Chunk,
        embedding: list[float],
        context: str = "",
    ) -> TextNode:
        metadata: dict[str, Any] = {
            **doc.meta,
            "filename": doc.filename,
            "parser": parser_name,
            "page": chunk.page,
            "section_path": chunk.section_path,
            "chunk_index": chunk.chunk_index,
            "job_id": job_id,
        }
        if context:
            metadata["context"] = context
        return TextNode(
            text=chunk.text,
            metadata=metadata,
            embedding=embedding,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc.id)},
        )

    async def _ensure_document_exists(self, document_id: str) -> None:
        """Raise DocumentDeletedError if the document was deleted while this job ran.

        Called right after the job writes something tied to the document (parse cache, vectors).
        Document deletion removes the record before the vectors and files, so either this check
        sees the record gone and the job removes its own output, or the record still existed and
        the deletion, which runs later, removes that output.
        """
        async with self.db.sessionmaker() as s:
            if await s.get(Document, document_id) is None:
                raise DocumentDeletedError

    async def _discard_output(self, job_id: str, ds: Dataset, document_id: str) -> None:
        """Remove what a job wrote for a document that no longer exists. The job row went with
        the document (FK cascade), so there is no status to update."""
        log.info("document %s was deleted during job %s; discarding its output", document_id, job_id)

        async def discard() -> None:
            await self.vectors.delete_job_nodes(ds, document_id, job_id)
            await asyncio.to_thread(self.blob.delete_parsed, ds.id, document_id)

        try:
            await asyncio.shield(discard())
        except asyncio.CancelledError:
            pass

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
