"""In-process job runner.

- asyncio.Queue + N worker tasks
- API keys live only in this object's memory while a job runs and are dropped afterwards
- On startup, queued/running jobs have no key anymore, so they are marked failed(interrupted)
"""

import asyncio
import contextlib
import logging

from sqlalchemy import update
from sqlmodel import col, select

from rag_service.db import Database
from rag_service.models.tables import (
    ACTIVE_JOB_STATUSES,
    Dataset,
    Document,
    DocumentStatus,
    IndexJob,
    JobStatus,
    utcnow,
)
from rag_service.services.indexing import IndexingPipeline
from rag_service.storage.vector import VectorStoreManager

log = logging.getLogger(__name__)


class JobRunner:
    def __init__(
        self,
        db: Database,
        pipeline: IndexingPipeline,
        vectors: VectorStoreManager,
        workers: int = 4,
    ) -> None:
        self.db = db
        self.pipeline = pipeline
        self.vectors = vectors
        self.n_workers = max(1, workers)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._keys: dict[str, str | None] = {}
        self._running: dict[str, asyncio.Task[None]] = {}
        self._user_canceled: set[str] = set()
        self._workers: list[asyncio.Task[None]] = []

    # ---------- lifecycle ----------

    async def start(self) -> None:
        await self.recover_interrupted()
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"job-worker-{i}")
            for i in range(self.n_workers)
        ]

    async def stop(self) -> None:
        for task in list(self._running.values()):
            task.cancel()
        if self._running:
            await asyncio.gather(*self._running.values(), return_exceptions=True)
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        self._keys.clear()

    async def recover_interrupted(self) -> None:
        async with self.db.sessionmaker() as s:
            jobs = (
                await s.exec(select(IndexJob).where(col(IndexJob.status).in_(ACTIVE_JOB_STATUSES)))
            ).all()
            for job in jobs:
                ds = await s.get(Dataset, job.dataset_id)
                if ds is not None:
                    with contextlib.suppress(Exception):
                        await self.vectors.delete_job_nodes(ds, job.document_id, job.id)
                job.status = JobStatus.failed
                job.error = "interrupted (server restarted); resubmit with API key"
                job.finished_at = utcnow()
                s.add(job)
                doc = await s.get(Document, job.document_id)
                if doc is not None and doc.status == DocumentStatus.indexing:
                    doc.status = DocumentStatus.indexed if doc.indexed_at else DocumentStatus.failed
                    doc.updated_at = utcnow()
                    s.add(doc)
            await s.commit()
            if jobs:
                log.warning("marked %d interrupted job(s) as failed", len(jobs))

    # ---------- submit / cancel ----------

    def submit(self, job_id: str, api_key: str | None) -> None:
        self._keys[job_id] = api_key
        self._queue.put_nowait(job_id)

    async def cancel(self, job_id: str) -> bool:
        """Queued jobs are canceled immediately; running jobs have their task canceled and are awaited until cleanup finishes.

        Returns: True if the cancellation was applied (False if the job had already finished).
        """
        async with self.db.sessionmaker() as s:
            res = await s.exec(
                update(IndexJob)
                .where(IndexJob.id == job_id, IndexJob.status == JobStatus.queued)
                .values(status=JobStatus.canceled, error="canceled", finished_at=utcnow())
            )
            await s.commit()
            if res.rowcount:
                self._keys.pop(job_id, None)
                await self._restore_document_status(job_id)
                return True

        task = self._running.get(job_id)
        if task is None:
            return False
        self._user_canceled.add(job_id)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        return True

    async def cancel_many(self, job_ids: list[str]) -> None:
        await asyncio.gather(*(self.cancel(j) for j in job_ids))

    async def _restore_document_status(self, job_id: str) -> None:
        async with self.db.sessionmaker() as s:
            job = await s.get(IndexJob, job_id)
            if job is None:
                return
            doc = await s.get(Document, job.document_id)
            if doc is None or doc.last_job_id != job_id:
                return
            if doc.status in (DocumentStatus.uploaded, DocumentStatus.indexing):
                doc.status = DocumentStatus.indexed if doc.indexed_at else DocumentStatus.uploaded
                doc.updated_at = utcnow()
                s.add(doc)
                await s.commit()

    # ---------- worker ----------

    async def _worker(self, n: int) -> None:
        while True:
            job_id = await self._queue.get()
            api_key = self._keys.get(job_id)
            task = asyncio.create_task(
                self.pipeline.run(
                    job_id, api_key, canceled_by_user=lambda j=job_id: j in self._user_canceled
                ),
                name=f"job-{job_id}",
            )
            self._running[job_id] = task
            try:
                await task
            except asyncio.CancelledError:
                if not task.cancelled() or job_id not in self._user_canceled:
                    # The worker itself was canceled (server shutdown)
                    task.cancel()
                    raise
            except Exception:  # noqa: BLE001
                log.exception("job %s crashed", job_id)
            finally:
                self._running.pop(job_id, None)
                self._keys.pop(job_id, None)
                self._user_canceled.discard(job_id)
                self._queue.task_done()

    async def wait_idle(self) -> None:
        """For tests: wait until every queued job has finished."""
        await self._queue.join()
