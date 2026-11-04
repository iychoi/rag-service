import asyncio
from pathlib import Path

from rag_service.parsers.base import DocumentParser


class ParserRegistry:
    """Name → parser, plus a per-parser concurrency limit."""

    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParser] = {}
        self._limits: dict[str, asyncio.Semaphore] = {}

    def register(self, parser: DocumentParser, concurrency: int) -> None:
        self._parsers[parser.name] = parser
        self._limits[parser.name] = asyncio.Semaphore(max(1, concurrency))

    def get(self, name: str) -> DocumentParser | None:
        return self._parsers.get(name)

    def limit(self, name: str) -> asyncio.Semaphore:
        return self._limits[name]

    def names(self) -> list[str]:
        return list(self._parsers)

    def all(self) -> list[DocumentParser]:
        return list(self._parsers.values())

    def resolve(self, requested: str, filename: str) -> DocumentParser:
        """Return the requested parser. md/txt files always use passthrough regardless of the request.

        Raises ValueError: unknown parser or unsupported file type.
        """
        ext = Path(filename).suffix.lower()
        passthrough = self._parsers.get("passthrough")
        if passthrough and ext in passthrough.supported_extensions:
            return passthrough
        parser = self._parsers.get(requested)
        if parser is None or parser.name == "passthrough":
            choices = [n for n in self._parsers if n != "passthrough"]
            raise ValueError(f"unknown parser '{requested}' (available: {choices})")
        if ext not in parser.supported_extensions:
            raise ValueError(
                f"parser '{parser.name}' does not support '{ext or 'no extension'}' files"
            )
        return parser
