"""Markdown → chunks.

1) Split into heading-level sections with MarkdownNodeParser → compute section_path
2) Re-split sections larger than chunk_size (tokens) with SentenceSplitter
3) Attach page numbers based on each chunk's position in the markdown

Sentence splitting uses a regex instead of NLTK punkt (English-only, needs a data download).
Separating whitespace is preserved so SentenceSplitter yields exact substrings of the source.
"""

import re
from dataclasses import dataclass, field

from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from rag_service.parsers.base import ParsedDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。？！])\s+|\n{2,}")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


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


def chunk_markdown(parsed: ParsedDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    markdown = parsed.markdown
    if not markdown.strip():
        return []
    sections = MarkdownNodeParser().get_nodes_from_documents([LIDocument(text=markdown)])
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, chunk_size // 2),
        chunking_tokenizer_fn=split_sentences,
    )

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
