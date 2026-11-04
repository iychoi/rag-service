"""mineru-api HTTP client. Targets MinerU 4.1.7.

Flow:
  POST /v1/tasks                              → task_id, file_id (waiting for upload)
  PUT  /v1/tasks/{task_id}/uploads/{file_id}  ← file bytes
  GET  /v1/tasks/{task_id}                    → waiting-file/pending/running → completed/partial/failed/canceled
  GET  /v1/files/{file_id}/content            → zip: markdown.md, structured_content.json, middle_json.json, images/

tier (MinerU 4.x quality/speed levels):
  flash    : text-layer extraction only (no models)
  basic    : lightweight local models (ONNX) — default on CPU
  standard : VLM (needs a local GPU or a remote VLM server)
  advanced : above standard
Non-PDF documents (Office, HTML, ...) are always parsed with flash regardless of tier.
"""

import asyncio
import contextlib
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx

from rag_service.parsers.base import (
    PageSpan,
    ParsedDocument,
    ParserError,
    ParserHealth,
    ParserUnavailableError,
)

TIERS = ("flash", "basic", "standard", "advanced")
OCR_MODES = ("auto", "txt", "ocr")
_TERMINAL = {"completed", "partial", "failed", "canceled"}
_NEEDLE_LEN = 40


def page_spans_from_structured(markdown: str, structured: dict[str, Any]) -> list[PageSpan] | None:
    """Build page spans by locating per-page block text from structured_content.json in the markdown.

    markdown.md has no page separators, so the position of each page's first block that appears verbatim
    in the markdown is treated as that page's start. Pages that cannot be located (e.g. image-only pages)
    are merged into the preceding page.
    """
    starts: list[tuple[int, int]] = []
    cursor = 0
    for i, page in enumerate(structured.get("pages") or []):
        page_no = int(page.get("page_idx", i)) + 1
        for block in page.get("blocks") or []:
            content = block.get("content")
            if not isinstance(content, str):
                continue
            needle = content.strip()[:_NEEDLE_LEN]
            if len(needle) < 2:
                continue
            pos = markdown.find(needle, cursor)
            if pos >= 0:
                # Snap to the line start so markdown markers before the text ("## ", "- ", ...) stay on the same page
                line_start = markdown.rfind("\n", 0, pos) + 1
                starts.append((page_no, max(line_start, starts[-1][1] + 1 if starts else 0)))
                cursor = pos + 1
                break
    if not starts:
        return None
    spans: list[PageSpan] = []
    for idx, (page_no, pos) in enumerate(starts):
        start = 0 if idx == 0 else pos
        end = starts[idx + 1][1] if idx + 1 < len(starts) else len(markdown)
        spans.append(PageSpan(page=page_no, start=start, end=end))
    return spans


class MinerUParser:
    name = "mineru"
    supported_extensions = frozenset(
        {
            # tiered (layout/OCR models)
            ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".jp2",
            # flash only (native documents)
            ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".rtf", ".odt", ".ods", ".odp",
            ".html", ".htm", ".mhtml", ".mht", ".csv", ".tsv", ".epub",
        }
    )  # fmt: skip

    def __init__(
        self,
        base_url: str,
        default_tier: str = "basic",
        timeout_s: float = 1800.0,
        poll_interval_s: float = 2.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if default_tier not in TIERS:
            raise ValueError(f"invalid MinerU tier '{default_tier}' (expected one of {TIERS})")
        self.base_url = base_url.rstrip("/")
        self.default_tier = default_tier
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=5.0),
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> ParserHealth:
        try:
            r = await self._client.get("/v1/health", timeout=5.0)
            r.raise_for_status()
            version = r.json().get("version")
            detail = None
            with contextlib.suppress(httpx.HTTPError, ValueError):
                t = await self._client.get("/v1/tiers", timeout=5.0)
                tiers = [x.get("id") for x in t.json().get("data", [])]
                detail = f"tiers: {', '.join(tiers)}; default tier: {self.default_tier}"
            return ParserHealth(available=True, version=version, detail=detail)
        except (httpx.HTTPError, ValueError) as e:
            return ParserHealth(available=False, detail=f"{type(e).__name__}: {e}")

    def validate_options(self, options: dict[str, Any]) -> dict[str, Any]:
        allowed = {"tier", "ocr_mode", "page_range"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"unsupported mineru options: {sorted(unknown)} (allowed: {sorted(allowed)})")
        if "tier" in options and options["tier"] not in TIERS:
            raise ValueError(f"mineru option 'tier' must be one of {TIERS}")
        if "ocr_mode" in options and options["ocr_mode"] not in OCR_MODES:
            raise ValueError(f"mineru option 'ocr_mode' must be one of {OCR_MODES}")
        if "page_range" in options and not isinstance(options["page_range"], str):
            raise ValueError("mineru option 'page_range' must be a string such as '1-5,8'")
        # Record the effective tier in the job options → part of the parse cache key (changing the default tier forces a re-parse)
        return {"tier": self.default_tier, **options}

    async def parse(self, path: Path, filename: str, options: dict[str, Any]) -> ParsedDocument:
        started = time.monotonic()
        data = await asyncio.to_thread(path.read_bytes)
        tier = options.get("tier", self.default_tier)
        body: dict[str, Any] = {
            "tier": tier,
            "ocr_mode": options.get("ocr_mode", "auto"),
            "output_formats": ["default"],
            "files": [
                {
                    "file_name": _mineru_file_name(filename),
                    "bytes": len(data),
                    "sha256sum": hashlib.sha256(data).hexdigest(),
                }
            ],
        }
        if options.get("page_range"):
            body["page_range"] = options["page_range"]

        task_id: str | None = None
        try:
            r = await self._client.post("/v1/tasks", json=body)
            _raise_for_status(r)
            created = r.json()
            task_id = created["task_id"]
            file_id = created["files"][0]["file_id"]

            r = await self._client.put(
                f"/v1/tasks/{task_id}/uploads/{file_id}",
                content=data,
                headers={"Content-Type": "application/octet-stream"},
            )
            _raise_for_status(r)

            file_info = await self._wait(task_id, started)
            r = await self._client.get(f"/v1/files/{file_id}/content")
            _raise_for_status(r)
            archive = r.content
        except asyncio.CancelledError:
            if task_id is not None:
                await self._cancel_quietly(task_id)
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise ParserUnavailableError(f"mineru service unreachable: {e}") from e
        except httpx.HTTPError as e:
            if task_id is not None:
                await self._cancel_quietly(task_id)
            raise ParserError(f"mineru request failed: {type(e).__name__}: {e}") from e

        markdown, structured = await asyncio.to_thread(_read_archive, archive)
        parse_info = file_info.get("parse") or {}
        return ParsedDocument(
            markdown=markdown,
            pages=page_spans_from_structured(markdown, structured) if structured else None,
            raw=structured,
            metadata={
                "parser": self.name,
                "backend": f"tier:{parse_info.get('tier', tier)}",
                "parser_version": parse_info.get("parser_version"),
                "duration_ms": parse_info.get("duration_ms"),
            },
        )

    async def _wait(self, task_id: str, started: float) -> dict[str, Any]:
        while True:
            if time.monotonic() - started > self.timeout_s:
                await self._cancel_quietly(task_id)
                raise ParserError(f"mineru parsing timed out after {self.timeout_s:.0f}s")
            r = await self._client.get(f"/v1/tasks/{task_id}")
            _raise_for_status(r)
            task = r.json()
            if task.get("status") in _TERMINAL:
                f = (task.get("files") or [{}])[0]
                if f.get("status") != "completed":
                    err = f.get("error") or {}
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise ParserError(
                        f"mineru parsing {f.get('status') or task.get('status')}: {msg or 'unknown error'}"
                    )
                return f
            await asyncio.sleep(self.poll_interval_s)

    async def _cancel_quietly(self, task_id: str) -> None:
        with contextlib.suppress(Exception):
            await asyncio.shield(self._client.post(f"/v1/tasks/{task_id}/cancel", timeout=5.0))


def _mineru_file_name(filename: str) -> str:
    """MinerU rejects '/', '\\', ':' and control characters in file names."""
    cleaned = "".join("_" if c in '/\\:' or ord(c) < 32 or ord(c) == 127 else c for c in filename)
    return cleaned if cleaned not in ("", ".", "..") else "document"


def _raise_for_status(r: httpx.Response) -> None:
    if r.is_success:
        return
    detail = ""
    with contextlib.suppress(ValueError, AttributeError):
        err = r.json().get("error") or r.json().get("detail")
        detail = err.get("message") if isinstance(err, dict) else str(err)
    raise ParserError(f"mineru service returned HTTP {r.status_code}: {detail or r.text[:300]}")


def _read_archive(archive: bytes) -> tuple[str, dict[str, Any] | None]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            names = set(z.namelist())
            if "markdown.md" not in names:
                raise ParserError(f"mineru result has no markdown.md (entries: {sorted(names)[:10]})")
            markdown = z.read("markdown.md").decode("utf-8")
            structured = None
            if "structured_content.json" in names:
                structured = json.loads(z.read("structured_content.json"))
            return markdown, structured
    except zipfile.BadZipFile as e:
        raise ParserError("mineru result is not a valid zip archive") from e
