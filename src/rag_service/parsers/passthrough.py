"""Markdown/text files are used as-is without going through a parser container."""

from pathlib import Path
from typing import Any

from rag_service.parsers.base import ParsedDocument, ParserError, ParserHealth


class PassthroughParser:
    name = "passthrough"
    supported_extensions = frozenset({".md", ".markdown", ".txt"})

    async def health(self) -> ParserHealth:
        return ParserHealth(available=True)

    def validate_options(self, options: dict[str, Any]) -> dict[str, Any]:
        if options:
            raise ValueError(f"passthrough parser takes no options (got {sorted(options)})")
        return {}

    async def parse(self, path: Path, filename: str, options: dict[str, Any]) -> ParsedDocument:
        raw = path.read_bytes()
        for enc in ("utf-8", "utf-8-sig", "cp949"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ParserError("could not decode text file (tried utf-8, cp949)")
        return ParsedDocument(markdown=text, metadata={"parser": self.name})
