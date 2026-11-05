"""Parsed document → chunks.

Two strategies, picked by what the parser left behind:

Structure (Docling, MinerU): the parser's structured output (raw.json) is turned into blocks in
reading order (headings, paragraphs, list items, tables) with exact page numbers.
- A heading starts a new chunk; the heading path becomes `section_path`. Headings with no text
  before the next heading are carried into the next chunk instead of becoming chunks of their own.
- Consecutive blocks are packed into one chunk while it stays within chunk_size tokens.
- A table is a chunk of its own, with its caption and footnotes. A table larger than chunk_size is
  cut between rows, and every piece repeats the header row and caption.
- A single block larger than chunk_size is split by sentence with chunk_overlap; packed blocks do
  not overlap, since they already end at paragraph boundaries.

Markdown (passthrough, or when no structure is available):
1) Split into heading-level sections with MarkdownNodeParser → compute section_path
2) Re-split sections larger than chunk_size (tokens) with SentenceSplitter
3) Attach page numbers based on each chunk's position in the markdown

Sentence splitting uses a regex instead of NLTK punkt (English-only, needs a data download).
Separating whitespace is preserved so SentenceSplitter yields exact substrings of the source.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from rag_service.parsers import docling, mineru
from rag_service.parsers.base import Block, ParsedDocument
from rag_service.services.tokens import count_tokens

log = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。？！])\s+|\n{2,}")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")

# Parser name → structured output (ParsedDocument.raw) to blocks
STRUCTURE_EXTRACTORS: dict[str, Callable[[dict[str, Any]], list[Block]]] = {
    "docling": docling.structure_blocks,
    "mineru": mineru.structure_blocks,
}


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    last = 0
    for m in _SENTENCE_BOUNDARY.finditer(text):
        out.append(text[last : m.end()])
        last = m.end()
    if last < len(text):
        out.append(text[last:])
    return out


@dataclass
class Chunk:
    text: str
    chunk_index: int
    section_path: list[str] = field(default_factory=list)
    page: int | None = None

    def embedding_text(self) -> str:
        """Embedding input: prefix the section path to add context."""
        if self.section_path:
            return " > ".join(self.section_path) + "\n\n" + self.text
        return self.text


def _splitter(chunk_size: int, chunk_overlap: int) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, chunk_size // 2),
        chunking_tokenizer_fn=split_sentences,
    )


def chunk_document(
    parsed: ParsedDocument, parser_name: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk by document structure when the parser provides it, otherwise by markdown."""
    extract = STRUCTURE_EXTRACTORS.get(parser_name)
    if extract is not None and parsed.raw:
        try:
            blocks = extract(parsed.raw)
        except Exception:  # noqa: BLE001 - an unexpected parser output must not fail the job
            log.warning("could not read %s structure; chunking markdown instead", parser_name,
                        exc_info=True)  # fmt: skip
            blocks = []
        if any(b.kind != "heading" for b in blocks):
            return chunk_blocks(blocks, chunk_size, chunk_overlap)
    return chunk_markdown(parsed, chunk_size, chunk_overlap)


# ---------- structure ----------


def chunk_blocks(blocks: list[Block], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    return _BlockChunker(chunk_size, chunk_overlap).run(blocks)


class _BlockChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.size = chunk_size
        self.splitter = _splitter(chunk_size, chunk_overlap)
        self.chunks: list[Chunk] = []
        self.headings: list[tuple[int, str]] = []  # (level, text) of the current heading path
        self.buf: list[str] = []  # markdown of blocks waiting to become a chunk
        self.buf_page: int | None = None
        self.buf_has_content = False  # False while buf holds only headings

    def run(self, blocks: list[Block]) -> list[Chunk]:
        for b in blocks:
            if b.kind == "heading":
                self._heading(b)
            elif b.kind == "table":
                self._table(b)
            else:
                self._text(b)
        self._flush()  # trailing headings with no text are kept rather than lost
        return self.chunks

    @property
    def path(self) -> list[str]:
        return [text for _, text in self.headings]

    def _emit(self, text: str, page: int | None) -> None:
        pieces = [text] if count_tokens(text) <= self.size else self.splitter.split_text(text)
        for piece in pieces:
            if piece.strip():
                self.chunks.append(Chunk(piece, len(self.chunks), self.path, page))

    def _flush(self) -> None:
        if self.buf:
            self._emit("\n\n".join(self.buf), self.buf_page)
        self.buf, self.buf_page, self.buf_has_content = [], None, False

    def _add(self, text: str, page: int | None, content: bool = True) -> None:
        if not self.buf:
            self.buf_page = page
        self.buf.append(text)
        self.buf_has_content = self.buf_has_content or content

    def _heading(self, b: Block) -> None:
        if self.buf_has_content:
            self._flush()
        while self.headings and self.headings[-1][0] >= b.level:
            self.headings.pop()
        self.headings.append((b.level, b.text))
        self._add(f"{'#' * b.level} {b.text}", b.page, content=False)

    def _text(self, b: Block) -> None:
        if self._fits([*self.buf, b.text]):
            self._add(b.text, b.page)
            return
        if self.buf_has_content:
            self._flush()
        if self._fits([*self.buf, b.text]):
            self._add(b.text, b.page)
            return
        # Too long even on its own: sentence-split it together with any pending headings
        page = self.buf_page if self.buf else b.page
        text = "\n\n".join([*self.buf, b.text])
        self.buf, self.buf_page, self.buf_has_content = [], None, False
        for piece in self.splitter.split_text(text):
            if piece.strip():
                self.chunks.append(Chunk(piece, len(self.chunks), self.path, page))

    def _table(self, b: Block) -> None:
        if self.buf_has_content:
            self._flush()
        prefix = "\n\n".join(self.buf)  # pending headings go with the table's first piece
        page = self.buf_page if self.buf else b.page
        self.buf, self.buf_page, self.buf_has_content = [], None, False

        budget = max(self.size - (count_tokens(prefix) + 1 if prefix else 0), self.size // 2)
        pieces = self._table_pieces(b, budget)
        for i, piece in enumerate(pieces):
            text = f"{prefix}\n\n{piece}" if prefix and i == 0 else piece
            self._emit(text, page if i == 0 else b.page)

    def _table_pieces(self, b: Block, budget: int) -> list[str]:
        whole = b.full_text()
        parts = b.table
        if parts is None or len(parts.rows) < 2 or count_tokens(whole) <= budget:
            return [whole]
        compose: Callable[[str], str] = b.full_text
        pieces: list[str] = []
        if count_tokens(compose(parts.render(parts.rows[:1]))) > budget:
            # Caption and footnotes leave no room for rows: give them their own piece and
            # repeat only the header row in the others
            pieces.append("\n\n".join(p for p in (b.caption, b.footnote) if p))
            compose = str
        rows: list[str] = []
        for row in parts.rows:
            if rows and count_tokens(compose(parts.render([*rows, row]))) > budget:
                pieces.append(compose(parts.render(rows)))
                rows = []
            rows.append(row)
        pieces.append(compose(parts.render(rows)))
        return pieces

    def _fits(self, texts: list[str]) -> bool:
        return count_tokens("\n\n".join(texts)) <= self.size


# ---------- markdown ----------


def chunk_markdown(parsed: ParsedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    markdown = parsed.markdown
    if not markdown.strip():
        return []
    sections = MarkdownNodeParser().get_nodes_from_documents([LIDocument(text=markdown)])
    splitter = _splitter(chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    cursor = 0
    for section in sections:
        text = section.get_content()
        if not text.strip():
            continue
        path = [p for p in str(section.metadata.get("header_path", "/")).split("/") if p]
        first_line = text.lstrip().split("\n", 1)[0]
        if m := _HEADING.match(first_line):
            path = [*path, m.group(1)]

        for piece in splitter.split_text(text):
            if not piece.strip():
                continue
            pos = markdown.find(piece[:64], cursor)
            if pos >= 0:
                cursor = pos + 1
            chunks.append(
                Chunk(
                    text=piece,
                    chunk_index=len(chunks),
                    section_path=path,
                    page=parsed.page_at(pos) if pos >= 0 else None,
                )
            )
    return chunks
