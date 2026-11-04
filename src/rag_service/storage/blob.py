"""Storage for original files and parse results. Local filesystem implementation.

Layout:
  {data_dir}/originals/{dataset_id}/{document_id}{ext}
  {data_dir}/parsed/{dataset_id}/{document_id}/{cache_key}/content.md | pages.json | meta.json | raw.json
"""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from rag_service.parsers.base import PageSpan, ParsedDocument

_CHUNK = 1024 * 1024


class UploadTooLargeError(Exception):
    pass


@dataclass
class StoredUpload:
    tmp_path: Path
    size: int
    sha256: str


class BlobStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir.resolve()
        self.originals = self.root / "originals"
        self.parsed = self.root / "parsed"
        self.tmp = self.root / "tmp"
        for d in (self.originals, self.parsed, self.tmp):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- originals ----------

    async def receive_upload(self, upload: UploadFile, max_bytes: int) -> StoredUpload:
        """Stream the upload to a temp file while enforcing the size limit and computing sha256."""
        fd, name = tempfile.mkstemp(dir=self.tmp, prefix="upload-")
        tmp_path = Path(name)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := await upload.read(_CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLargeError
                    digest.update(chunk)
                    out.write(chunk)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        return StoredUpload(tmp_path=tmp_path, size=size, sha256=digest.hexdigest())

    def original_path(self, dataset_id: str, document_id: str, filename: str) -> Path:
        ext = Path(filename).suffix.lower()
        if not ext.isascii() or len(ext) > 16:
            ext = ""
        return self.originals / dataset_id / f"{document_id}{ext}"

    def commit_upload(self, stored: StoredUpload, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stored.tmp_path, dest)

    def discard_upload(self, stored: StoredUpload) -> None:
        stored.tmp_path.unlink(missing_ok=True)

    def resolve(self, storage_path: str) -> Path:
        path = (self.root / storage_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("storage path escapes data dir")
        return path

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    # ---------- parse cache ----------

    @staticmethod
    def parse_cache_key(parser: str, options: dict[str, Any]) -> str:
        opts = json.dumps(options, sort_keys=True, separators=(",", ":"))
        return f"{parser}-{hashlib.sha256(opts.encode()).hexdigest()[:12]}"

    def _parsed_dir(self, dataset_id: str, document_id: str, cache_key: str) -> Path:
        return self.parsed / dataset_id / document_id / cache_key

    def load_parsed(self, dataset_id: str, document_id: str, cache_key: str) -> ParsedDocument | None:
        d = self._parsed_dir(dataset_id, document_id, cache_key)
        md = d / "content.md"
        if not md.exists():
            return None
        pages_file = d / "pages.json"
        meta_file = d / "meta.json"
        pages = None
        if pages_file.exists():
            pages = [PageSpan.model_validate(p) for p in json.loads(pages_file.read_text())]
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        return ParsedDocument(markdown=md.read_text(encoding="utf-8"), pages=pages, metadata=meta)

    def save_parsed(
        self, dataset_id: str, document_id: str, cache_key: str, parsed: ParsedDocument
    ) -> None:
        d = self._parsed_dir(dataset_id, document_id, cache_key)
        d.mkdir(parents=True, exist_ok=True)
        if parsed.raw is not None:
            (d / "raw.json").write_text(json.dumps(parsed.raw, ensure_ascii=False))
        if parsed.pages is not None:
            (d / "pages.json").write_text(json.dumps([p.model_dump() for p in parsed.pages]))
        (d / "meta.json").write_text(json.dumps(parsed.metadata, ensure_ascii=False, default=str))
        # Write content.md last and atomically; its presence marks the cache entry as complete
        tmp = d / "content.md.tmp"
        tmp.write_text(parsed.markdown, encoding="utf-8")
        os.replace(tmp, d / "content.md")

    def latest_parsed_markdown(self, dataset_id: str, document_id: str, cache_key: str) -> str | None:
        md = self._parsed_dir(dataset_id, document_id, cache_key) / "content.md"
        return md.read_text(encoding="utf-8") if md.exists() else None

    # ---------- deletion ----------

    def delete_document(self, dataset_id: str, document_id: str, storage_path: str) -> None:
        self.resolve(storage_path).unlink(missing_ok=True)
        shutil.rmtree(self.parsed / dataset_id / document_id, ignore_errors=True)

    def delete_dataset(self, dataset_id: str) -> None:
        shutil.rmtree(self.originals / dataset_id, ignore_errors=True)
        shutil.rmtree(self.parsed / dataset_id, ignore_errors=True)
