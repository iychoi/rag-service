"""Common parser interface."""

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ParserError(Exception):
    """Parsing failed. The message may be shown to users, so never include sensitive data."""


class ParserUnavailableError(ParserError):
    """The parser service is unreachable."""


class PageSpan(BaseModel):
    """[start, end) character offsets of one page within the markdown."""

    page: int
    start: int
    end: int


class ParsedDocument(BaseModel):
    markdown: str
    pages: list[PageSpan] | None = None
    raw: dict[str, Any] | None = None  # parser-specific structure (DoclingDocument JSON, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def page_at(self, offset: int) -> int | None:
        if not self.pages:
            return None
        for span in self.pages:
            if offset < span.end:
                return span.page
        return self.pages[-1].page


class ParserHealth(BaseModel):
    available: bool
    detail: str | None = None
    version: str | None = None


class DocumentParser(Protocol):
    name: str
    supported_extensions: frozenset[str]

    async def health(self) -> ParserHealth: ...

    async def parse(self, path: Path, filename: str, options: dict[str, Any]) -> ParsedDocument: ...

    def validate_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Validate/normalize request options. Raises ValueError for invalid options."""
        ...


def split_pages(markdown: str, placeholder: str) -> tuple[str, list[PageSpan] | None]:
    """Split markdown on the page separator, compute page offsets, and return the markdown without separators."""
    if not placeholder or placeholder not in markdown:
        return markdown, None
    parts = markdown.split(placeholder)
    out: list[str] = []
    spans: list[PageSpan] = []
    pos = 0
    for i, part in enumerate(parts):
        text = part.strip("\n")
        if i > 0:
            out.append("\n\n")
            pos += 2
        spans.append(PageSpan(page=i + 1, start=pos, end=pos + len(text)))
        out.append(text)
        pos += len(text)
    return "".join(out), spans
