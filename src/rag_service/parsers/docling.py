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
    Block,
    ParsedDocument,
    ParserError,
    ParserHealth,
    ParserUnavailableError,
    split_pages,
)
from rag_service.parsers.tables import markdown_table

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


# ---------- structure: DoclingDocument JSON → blocks ----------

_SKIP_LABELS = {"page_header", "page_footer"}


def structure_blocks(doc: dict[str, Any]) -> list[Block]:
    """Blocks in reading order from a DoclingDocument (the `json_content` cached as raw.json).

    Walks `body` depth first. Only the body content layer is used (page headers and footers live in
    `furniture`). Tables become one block with their captions and footnotes; pictures contribute
    their captions only. Returns [] if `doc` is not a DoclingDocument.
    """
    if doc.get("schema_name") != "DoclingDocument" or not isinstance(doc.get("body"), dict):
        return []
    blocks: list[Block] = []

    def resolve(ref: Any) -> dict[str, Any] | None:
        path = ref.get("$ref") if isinstance(ref, dict) else None
        if not isinstance(path, str) or not path.startswith("#/"):
            return None
        node: Any = doc
        for part in path[2:].split("/"):
            if isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node if isinstance(node, dict) else None

    def page(item: dict[str, Any]) -> int | None:
        prov = item.get("prov") or []
        return prov[0].get("page_no") if prov and isinstance(prov[0], dict) else None

    def texts_of(refs: Any) -> list[str]:
        items = (resolve(r) for r in refs or [])
        return [t for i in items if i and (t := (i.get("text") or "").strip())]

    def inline_text(item: dict[str, Any]) -> str:
        if text := (item.get("text") or "").strip():
            return text
        children = (resolve(r) for r in item.get("children") or [])
        return " ".join(t for c in children if c and (t := inline_text(c)))

    def visit(item: dict[str, Any]) -> None:
        if item.get("content_layer", "body") != "body":
            return
        ref = str(item.get("self_ref", ""))
        label = item.get("label")
        if ref.startswith("#/texts/"):
            add_text(item)
        elif ref.startswith("#/tables/"):
            add_table(item)
            return  # children are its captions/footnotes
        elif ref.startswith("#/pictures/"):
            if captions := texts_of(item.get("captions")):
                blocks.append(Block("text", "\n".join(captions), page(item)))
            return
        elif ref.startswith("#/groups/") and label == "inline":
            if text := inline_text(item):
                blocks.append(Block("text", text, page(item) or first_page(item)))
            return
        for child_ref in item.get("children") or []:
            if (child := resolve(child_ref)) is not None:
                visit(child)

    def first_page(item: dict[str, Any]) -> int | None:
        for child in (resolve(r) for r in item.get("children") or []):
            if child and (p := page(child) or first_page(child)):
                return p
        return None

    def add_text(item: dict[str, Any]) -> None:
        label = item.get("label")
        text = (item.get("text") or "").strip()
        if label in _SKIP_LABELS or not text:
            return
        if label == "title":
            blocks.append(Block("heading", text, page(item), level=1))
        elif label == "section_header":
            level = min(int(item.get("level") or 1) + 1, 6)  # same as Docling's markdown: level 1 → ##
            blocks.append(Block("heading", text, page(item), level=level))
        elif label == "list_item":
            marker = item.get("marker") or "-"
            blocks.append(Block("text", f"{marker} {text}", page(item)))
        elif label == "code":
            blocks.append(Block("text", f"```\n{text}\n```", page(item)))
        else:
            blocks.append(Block("text", text, page(item)))

    def add_table(item: dict[str, Any]) -> None:
        caption = "\n".join(texts_of(item.get("captions")))
        footnote = "\n".join(texts_of(item.get("footnotes")))
        grid = (item.get("data") or {}).get("grid") or []
        rows = [[str(c.get("text") or "") for c in row] for row in grid]
        header_rows = 0
        for row in grid:
            if not any(c.get("column_header") for c in row):
                break
            header_rows += 1
        parts = markdown_table(rows, header_rows)
        if parts is None:
            if caption or footnote:
                blocks.append(Block("text", "\n\n".join(p for p in (caption, footnote) if p), page(item)))
            return
        blocks.append(
            Block("table", parts.render(), page(item), caption=caption, footnote=footnote, table=parts)
        )

    visit(doc["body"])
    return blocks
