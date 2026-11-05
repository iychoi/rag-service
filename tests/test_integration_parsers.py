"""Integration tests against real parser containers (docling-serve, mineru-api).

Run: DOCLING_URL=http://localhost:5001 MINERU_URL=http://localhost:8010 pytest -m integration
"""

import os
from pathlib import Path

import pytest

from rag_service.parsers.base import ParsedDocument
from rag_service.parsers.docling import DoclingParser
from rag_service.services.chunking import chunk_document, chunk_markdown
from tests.conftest import FIXTURES

pytestmark = pytest.mark.integration

DOCLING_URL = os.environ.get("DOCLING_URL", "http://localhost:5001")


async def test_real_docling_parses_pdf_with_pages() -> None:
    parser = DoclingParser(DOCLING_URL, timeout_s=300)
    try:
        health = await parser.health()
        if not health.available:
            pytest.skip(f"docling not reachable at {DOCLING_URL}: {health.detail}")
        parsed = await parser.parse(FIXTURES / "sample.pdf", "sample.pdf", {"do_ocr": False})
    finally:
        await parser.aclose()

    assert "Termination" in parsed.markdown
    assert parsed.pages is not None and len(parsed.pages) == 2
    assert parsed.raw is not None and parsed.raw.get("schema_name") == "DoclingDocument"

    chunks = chunk_markdown(parsed, chunk_size=256, chunk_overlap=32)
    term = next(c for c in chunks if "terminate" in c.text)
    assert term.page == 2
    assert term.section_path == ["Article 5. Termination"]


MINERU_URL = os.environ.get("MINERU_URL", "http://localhost:8010")


async def test_real_mineru_parses_pdf_with_pages() -> None:
    from rag_service.parsers.mineru import MinerUParser

    parser = MinerUParser(MINERU_URL, default_tier="basic", timeout_s=600, poll_interval_s=1)
    try:
        health = await parser.health()
        if not health.available:
            pytest.skip(f"mineru not reachable at {MINERU_URL}: {health.detail}")
        parsed = await parser.parse(
            FIXTURES / "sample.pdf", "sample.pdf", parser.validate_options({})
        )
    finally:
        await parser.aclose()

    assert "Termination" in parsed.markdown
    assert parsed.metadata["backend"] == "tier:basic"
    assert parsed.pages is not None and [p.page for p in parsed.pages] == [1, 2]
    chunks = chunk_markdown(parsed, chunk_size=256, chunk_overlap=32)
    term = next(c for c in chunks if "terminate" in c.text)
    assert term.page == 2 and term.section_path == ["Article 5. Termination"]


async def _parse_with(name: str, path: Path) -> ParsedDocument:
    from rag_service.parsers.mineru import MinerUParser

    parser: DoclingParser | MinerUParser
    if name == "docling":
        parser = DoclingParser(DOCLING_URL, timeout_s=300)
    else:
        parser = MinerUParser(MINERU_URL, default_tier="basic", timeout_s=600, poll_interval_s=1)
    try:
        health = await parser.health()
        if not health.available:
            pytest.skip(f"{name} not reachable: {health.detail}")
        return await parser.parse(path, path.name, parser.validate_options({}))
    finally:
        await parser.aclose()


@pytest.mark.parametrize("name", ["docling", "mineru"])
async def test_real_parser_structure_keeps_table_whole(name: str) -> None:
    """table.pdf: 2 pages, 4 sections, a 3-row price table on page 1. The recorded outputs in
    tests/fixtures/table_{docling,mineru}.json come from this test's parsers."""
    parsed = await _parse_with(name, FIXTURES / "table.pdf")
    chunks = chunk_document(parsed, name, chunk_size=512, chunk_overlap=64)
    assert [(c.section_path, c.page) for c in chunks] == [
        (["1. Scope"], 1), (["2. Price Table"], 1), (["3. Termination"], 2), (["4. Support"], 2),
    ]  # fmt: skip
    table = chunks[1].text
    assert "Table 1. Monthly prices by region." in table
    for region in ("us-east", "eu-west", "ap-north"):
        assert region in table
