"""docling-serve HTTP client.

Uses the async API of docling-serve 1.38:
  POST /v1/convert/file/async  → task_id
  GET  /v1/status/poll/{id}    → task_status (pending/started/success/partial_success/failure/skipped)
  GET  /v1/result/{id}         → ConvertDocumentResponse | TaskFailureResult
"""

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from rag_service.parsers.base import (
    ParsedDocument,
    ParserError,
    ParserHealth,
    ParserUnavailableError,
    split_pages,
)

PAGE_BREAK = "<!-- rag-service:page-break -->"

_DONE = {"success", "partial_success", "failure", "skipped"}

# Options (and their types) accepted from requests; anything else is rejected
_ALLOWED_OPTIONS: dict[str, type | tuple[type, ...]] = {
    "do_ocr": bool,
    "force_ocr": bool,
    "ocr_engine": str,
    "ocr_lang": list,
    "do_table_structure": bool,
    "table_mode": str,
    "pdf_backend": str,
    "document_timeout": (int, float),
}


class DoclingParser:
    name = "docling"
    supported_extensions = frozenset(
        {
            ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".xhtml", ".csv", ".adoc",
            ".asciidoc", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
        }
    )  # fmt: skip

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 900.0,
        poll_interval_s: float = 2.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url, timeout=httpx.Timeout(60.0, connect=5.0)
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> ParserHealth:
        try:
            r = await self._client.get("/health", timeout=5.0)
            r.raise_for_status()
            version = None
            try:
                v = await self._client.get("/version", timeout=5.0)
                version = v.json().get("docling-serve")
            except (httpx.HTTPError, ValueError):
                pass
            return ParserHealth(available=True, version=version)
        except httpx.HTTPError as e:
            return ParserHealth(available=False, detail=f"{type(e).__name__}: {e}")

    def validate_options(self, options: dict[str, Any]) -> dict[str, Any]:
        unknown = set(options) - set(_ALLOWED_OPTIONS)
        if unknown:
            raise ValueError(
                f"unsupported docling options: {sorted(unknown)} "
                f"(allowed: {sorted(_ALLOWED_OPTIONS)})"
            )
        for key, value in options.items():
            if not isinstance(value, _ALLOWED_OPTIONS[key]) or (
                key == "document_timeout" and isinstance(value, bool)
            ):
                raise ValueError(f"docling option '{key}' has invalid type")
        if "ocr_lang" in options and not all(isinstance(x, str) for x in options["ocr_lang"]):
            raise ValueError("docling option 'ocr_lang' must be a list of strings")
        return dict(options)

    def _form(self, options: dict[str, Any]) -> dict[str, str | list[str]]:
        form: dict[str, str | list[str]] = {
            "to_formats": ["md", "json"],
            "md_page_break_placeholder": PAGE_BREAK,
            "image_export_mode": "placeholder",
        }
        for key, value in options.items():
            if isinstance(value, list):
                form[key] = [str(v) for v in value]
            elif isinstance(value, bool):
                form[key] = "true" if value else "false"
            else:
                form[key] = str(value)
        return form

    async def parse(self, path: Path, filename: str, options: dict[str, Any]) -> ParsedDocument:
        started = time.monotonic()
        try:
            with path.open("rb") as fh:
                r = await self._client.post(
                    "/v1/convert/file/async",
                    data=self._form(options),
                    files={"files": (filename, fh, "application/octet-stream")},
                )
            r.raise_for_status()
            task_id = r.json()["task_id"]

            status = await self._wait(task_id, started)
            r = await self._client.get(f"/v1/result/{task_id}")
            r.raise_for_status()
            result = r.json()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise ParserUnavailableError(f"docling service unreachable: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ParserError(
                f"docling service returned HTTP {e.response.status_code}: {e.response.text[:500]}"
            ) from e
        except httpx.HTTPError as e:
            raise ParserError(f"docling request failed: {type(e).__name__}: {e}") from e

        if "failure" in result and "document" not in result:
            raise ParserError(f"docling conversion failed: {result['failure']}")
        if status not in ("success", "partial_success") or result.get("status") not in (
            "success",
            "partial_success",
        ):
            raise ParserError(
                f"docling conversion ended with status '{result.get('status', status)}': "
                f"{result.get('errors')}"
            )

        doc = result.get("document") or {}
        md = doc.get("md_content")
        if md is None:
            raise ParserError("docling returned no markdown content")
        markdown, pages = split_pages(md, PAGE_BREAK)
        return ParsedDocument(
            markdown=markdown,
            pages=pages,
            raw=doc.get("json_content"),
            metadata={
                "parser": self.name,
                "status": result.get("status"),
                "errors": result.get("errors") or [],
                "processing_time": result.get("processing_time"),
                "confidence": (result.get("confidence") or {}).get("mean_grade"),
            },
        )

    async def _wait(self, task_id: str, started: float) -> str:
        while True:
            remaining = self.timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                raise ParserError(f"docling conversion timed out after {self.timeout_s:.0f}s")
            r = await self._client.get(
                f"/v1/status/poll/{task_id}", params={"wait": min(5.0, remaining)}
            )
            r.raise_for_status()
            body = r.json()
            status = body.get("task_status")
            if status in _DONE:
                if status == "failure":
                    raise ParserError(
                        f"docling conversion failed: {body.get('error_message') or body.get('failure')}"
                    )
                return status
            await asyncio.sleep(self.poll_interval_s)
