"""Structure-aware chunking from Docling / MinerU structured output (recorded from the real
parsers on tests/fixtures/table.pdf), and the markdown fallback."""

import json
from pathlib import Path

from rag_service.parsers import docling, mineru
from rag_service.parsers.base import Block, ParsedDocument
from rag_service.parsers.tables import html_table_parts, markdown_table, markdown_table_parts
from rag_service.services.chunking import Chunk, chunk_blocks, chunk_document
from rag_service.services.tokens import count_tokens
from rag_service.storage.blob import BlobStore
from tests.conftest import FIXTURES

TABLE_ROWS = ["| us-east | 0.023 USD | 0.09 USD |", "| eu-west | 0.025 USD | 0.10 USD |",
              "| ap-north | 0.027 USD | 0.12 USD |"]  # fmt: skip


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def summary(chunks: list[Chunk]) -> list[tuple[list[str], int | None]]:
    return [(c.section_path, c.page) for c in chunks]


# ---------------- extraction ----------------


def test_docling_blocks() -> None:
    blocks = docling.structure_blocks(load("table_docling.json"))
    assert [(b.kind, b.page) for b in blocks][:5] == [
        ("heading", 1), ("heading", 1), ("text", 1), ("heading", 1), ("table", 1),
    ]  # fmt: skip
    assert blocks[1].text == "1. Scope" and blocks[1].level == 2  # section_header level 1 → ##
    table = blocks[4]
    assert table.caption == "Table 1. Monthly prices by region."
    assert table.table is not None and table.table.rows == TABLE_ROWS
    assert table.table.head[0] == "| Region | Storage per GB | Egress per GB |"
    assert [b.text for b in blocks[-2:]] == [
        "- Email support on business days.", "- Phone support for enterprise plans.",
    ]  # fmt: skip
    # Not a DoclingDocument (e.g. the trimmed fixture used by the HTTP client tests)
    assert docling.structure_blocks(load("docling_result.json")["document"]["json_content"]) == []


def test_mineru_blocks() -> None:
    blocks = mineru.structure_blocks(load("table_mineru.json"))
    assert [(b.kind, b.page) for b in blocks][:5] == [
        ("heading", 1), ("heading", 1), ("text", 1), ("heading", 1), ("table", 1),
    ]  # fmt: skip
    table = blocks[4]
    assert table.caption == "Table 1. Monthly prices by region."
    assert table.table is not None and len(table.table.rows) == 3
    assert blocks[5].kind == "heading" and blocks[5].page == 2
    assert mineru.structure_blocks({"no": "pages"}) == []


def test_mineru_skips_furniture_and_image_links() -> None:
    raw = {"pages": [{"page_idx": 0, "blocks": [
        {"type": "header", "content": "ACME Corp confidential"},
        {"type": "text", "content": "Body text."},
        {"type": "image", "content": "![](images/a.jpg)", "captions": [{"content": "Figure 1. Layout."}]},
        {"type": "page_number", "content": "1"},
    ]}]}  # fmt: skip
    assert [(b.kind, b.text) for b in mineru.structure_blocks(raw)] == [
        ("text", "Body text."), ("text", "Figure 1. Layout."),
    ]  # fmt: skip


def test_table_parts() -> None:
    parts = markdown_table_parts("| a | b |\n|---|:-:|\n| 1 | 2 |\n| 3 | 4 |")
    assert parts is not None and parts.head == ["| a | b |", "|---|:-:|"] and len(parts.rows) == 2
    assert markdown_table_parts("just | a line") is None

    html = "<table><thead><tr><th>a</th></tr></thead><tbody><tr><td>1</td></tr><tr><td>2</td></tr></tbody></table>"
    parts = html_table_parts(html)
    assert parts is not None and parts.head == ["<tr><th>a</th></tr>"]
    assert parts.render(parts.rows[1:]) == "<table><tr><th>a</th></tr><tr><td>2</td></tr></table>"

    parts = markdown_table([["h1", "h2"], ["sub", ""], ["x|y", "z"]], header_rows=2)
    assert parts is not None
    assert parts.head == ["| h1 | h2 |", "| --- | --- |", "| sub |  |"]
    assert parts.rows == ["| x\\|y | z |"]


# ---------------- chunking ----------------


def test_both_parsers_chunk_the_same_way() -> None:
    expected = [(["1. Scope"], 1), (["2. Price Table"], 1), (["3. Termination"], 2),
                (["4. Support"], 2)]  # fmt: skip
    for name, raw in (("docling", load("table_docling.json")), ("mineru", load("table_mineru.json"))):
        chunks = chunk_document(ParsedDocument(markdown="unused", raw=raw), name, 512, 64)
        assert summary(chunks) == expected, name
        assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
        # A heading with no text of its own travels with the next chunk
        assert chunks[0].text.startswith("## Service Level Agreement\n\n## 1. Scope\n\nThis agreement")
        # The table stays whole, with its heading and caption
        table = chunks[1].text
        assert table.startswith("## 2. Price Table\n\nTable 1. Monthly prices by region.\n\n|")
        assert all(row.split("|")[1].strip() in table for row in TABLE_ROWS), name


def test_blocks_are_packed_up_to_chunk_size() -> None:
    para = "The service stores data in three regions. " * 6
    blocks = [Block("heading", "Terms", 1, level=1)] + [Block("text", para, 1) for _ in range(8)]
    chunks = chunk_blocks(blocks, chunk_size=128, chunk_overlap=16)
    assert 1 < len(chunks) < 8
    assert all(count_tokens(c.text) <= 128 for c in chunks)
    assert all(c.section_path == ["Terms"] for c in chunks)
    # Packed paragraphs are whole: no paragraph is cut or repeated across chunks
    assert sum(c.text.count(para.strip()) for c in chunks) == 8


def test_long_block_is_sentence_split_with_overlap() -> None:
    long = " ".join(f"Sentence number {i} explains a clause." for i in range(60))
    chunks = chunk_blocks([Block("heading", "A", 3, level=2), Block("text", long, 3)], 64, 16)
    assert len(chunks) > 3
    assert all(count_tokens(c.text) <= 64 and c.page == 3 for c in chunks)
    assert chunks[0].text.startswith("## A\n\nSentence number 0")
    assert "Sentence number 59" in chunks[-1].text


def test_large_table_is_split_between_rows() -> None:
    rows = [[f"region-{i}", f"0.{i:03d} USD", f"tier {i % 3}"] for i in range(60)]
    parts = markdown_table([["Region", "Price", "Tier"], *rows], header_rows=1)
    assert parts is not None
    table = Block("table", parts.render(), 4, caption="Table 9. Prices.", table=parts)
    blocks = [Block("heading", "Prices", 4, level=1), table, Block("text", "After the table.", 5)]
    chunks = chunk_blocks(blocks, chunk_size=128, chunk_overlap=16)

    table_chunks = chunks[:-1]
    assert len(table_chunks) > 2
    assert table_chunks[0].text.startswith("# Prices\n\nTable 9. Prices.\n\n| Region |")
    for c in table_chunks:
        assert count_tokens(c.text) <= 128
        assert "Table 9. Prices.\n\n| Region | Price | Tier |\n| --- | --- | --- |" in c.text
    seen = [line for c in table_chunks for line in c.text.splitlines() if "region-" in line]
    assert seen == parts.rows  # every row exactly once, in order
    assert chunks[-1].text == "After the table." and chunks[-1].page == 5  # not merged into the table


def test_falls_back_to_markdown() -> None:
    md = "# Title\n\nSome text."
    expected = [(["Title"], None)]
    # passthrough has no structure; a raw that is not recognized; a raw that breaks the extractor
    assert summary(chunk_document(ParsedDocument(markdown=md), "passthrough", 256, 32)) == expected
    assert summary(chunk_document(ParsedDocument(markdown=md, raw={"x": 1}), "docling", 256, 32)) == expected
    broken = {"pages": [{"page_idx": "not a number", "blocks": []}]}
    assert summary(chunk_document(ParsedDocument(markdown=md, raw=broken), "mineru", 256, 32)) == expected


def test_parse_cache_keeps_structure(tmp_path: Path) -> None:
    blob = BlobStore(tmp_path)
    raw = load("table_mineru.json")
    blob.save_parsed("ds", "doc", "mineru-abc", ParsedDocument(markdown="# x", raw=raw))
    cached = blob.load_parsed("ds", "doc", "mineru-abc")
    assert cached is not None and cached.raw == raw
