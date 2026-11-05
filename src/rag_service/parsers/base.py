"""Common parser interface."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

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


@dataclass
class TableParts:
    """A table split into rows, so an oversized table can be cut between rows.

    `head` (header row and markdown separator, or an HTML header row) is repeated in every piece.
    """

    head: list[str]
    rows: list[str]
    open: str = ""  # e.g. "<table>" for HTML tables
    close: str = ""
    sep: str = "\n"

    def render(self, rows: list[str] | None = None) -> str:
        body = self.rows if rows is None else rows
        return self.open + self.sep.join([*self.head, *body]) + self.close


@dataclass
class Block:
    """One item of a document in reading order, taken from a parser's structured output.

    `text` is markdown. Headings carry their `level` (1 = top). Tables keep their caption and
    footnotes separately so every piece of a split table can repeat them.
    """

    kind: Literal["heading", "text", "table"]
    text: str
    page: int | None = None
    level: int = 1
    caption: str = ""
    footnote: str = ""
    table: TableParts | None = field(default=None, repr=False)

    def full_text(self, table_text: str | None = None) -> str:
        if self.kind != "table":
            return self.text
        body = self.text if table_text is None else table_text
        return "\n\n".join(p for p in (self.caption, body, self.footnote) if p)


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
