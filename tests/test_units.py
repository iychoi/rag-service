"""Unit tests without a DB: page splitting, chunking, hash embedding, parser selection."""

import math

import pytest

from rag_service.parsers.base import ParsedDocument, split_pages
from rag_service.parsers.docling import DoclingParser
from rag_service.parsers.passthrough import PassthroughParser
from rag_service.parsers.registry import ParserRegistry
from rag_service.providers.embeddings import HashEmbedding
from rag_service.services.chunking import chunk_markdown, split_sentences


def test_split_pages_maps_offsets() -> None:
    md, pages = split_pages("p1 text\n\nPB\n\np2 text\n\nPB\n\np3", "PB")
    assert "PB" not in md
    assert pages is not None and [p.page for p in pages] == [1, 2, 3]
    doc = ParsedDocument(markdown=md, pages=pages)
    assert doc.page_at(md.index("p2")) == 2
    assert doc.page_at(md.index("p3")) == 3
    assert split_pages("no breaks", "PB") == ("no breaks", None)


def test_split_sentences_preserves_text() -> None:
    text = "첫 문장이다. 둘째 문장! 셋째?\n\n다음 문단."
    assert "".join(split_sentences(text)) == text
    assert len(split_sentences(text)) == 4


def test_chunk_markdown_sections_and_pages() -> None:
    body = "해지 조건은 다음과 같다. " * 80
    md = f"# 약관\n\n개요.\n\n## 제5조 해지\n\n{body}\n\nPB\n\n## 제7조 배상\n\n배상 규정."
    md, pages = split_pages(md, "PB")
    chunks = chunk_markdown(ParsedDocument(markdown=md, pages=pages), chunk_size=128, chunk_overlap=16)
    assert len(chunks) > 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    sec5 = [c for c in chunks if c.section_path == ["약관", "제5조 해지"]]
    assert len(sec5) > 1  # a long section is split into several chunks sharing one section_path
    last = chunks[-1]
    assert last.section_path == ["약관", "제7조 배상"]
    assert last.page == 2
    assert chunks[0].page == 1
    assert last.embedding_text().startswith("약관 > 제7조 배상\n\n")


def test_chunk_markdown_empty() -> None:
    assert chunk_markdown(ParsedDocument(markdown="  \n"), 256, 32) == []


def test_hash_embedding_similarity() -> None:
    emb = HashEmbedding(model_name="t", dim=512)

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    q = emb.get_query_embedding("계약 해지 조건")
    near = emb.get_text_embedding("계약을 해지할 수 있는 조건은 다음과 같다")
    far = emb.get_text_embedding("weekly meeting budget approval")
    assert math.isclose(sum(x * x for x in q), 1.0, rel_tol=1e-6)
    assert cos(q, near) > cos(q, far)
    assert emb.get_text_embedding("") == emb.get_text_embedding("   ")  # constant vector instead of a zero vector


def test_registry_resolution() -> None:
    reg = ParserRegistry()
    reg.register(PassthroughParser(), 1)
    reg.register(DoclingParser("http://x"), 1)
    assert reg.resolve("docling", "a.md").name == "passthrough"
    assert reg.resolve("docling", "a.PDF").name == "docling"
    with pytest.raises(ValueError, match="unknown parser"):
        reg.resolve("mineru", "a.pdf")
    with pytest.raises(ValueError, match="does not support"):
        reg.resolve("docling", "a.exe")
    with pytest.raises(ValueError, match="unknown parser"):
        reg.resolve("passthrough", "a.pdf")


def test_docling_option_validation() -> None:
    p = DoclingParser("http://x")
    assert p.validate_options({"do_ocr": False, "ocr_lang": ["ko", "en"]}) == {
        "do_ocr": False,
        "ocr_lang": ["ko", "en"],
    }
    with pytest.raises(ValueError, match="unsupported"):
        p.validate_options({"to_formats": "html"})
    with pytest.raises(ValueError, match="invalid type"):
        p.validate_options({"do_ocr": "yes"})
