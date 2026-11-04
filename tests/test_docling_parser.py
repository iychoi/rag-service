"""docling-serve HTTP client (responses are recordings of real docling-serve 1.38.2 output)."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from rag_service.parsers.base import ParserError, ParserUnavailableError
from rag_service.parsers.docling import PAGE_BREAK, DoclingParser

FIXTURES = Path(__file__).parent / "fixtures"
URL = "http://docling.unit"


def recorded_result() -> dict:
    data = json.loads((FIXTURES / "docling_result.json").read_text())
    md = data["document"]["md_content"]
    data["document"]["md_content"] = md.replace("<!-- page-break -->", PAGE_BREAK)
    return data


def mock_docling(router: respx.MockRouter, result: dict, statuses: list[str]) -> respx.Route:
    submit = router.post(f"{URL}/v1/convert/file/async").respond(
        json={"task_id": "t1", "task_type": "convert", "task_status": "pending"}
    )
    router.get(f"{URL}/v1/status/poll/t1").mock(
        side_effect=[
            httpx.Response(200, json={"task_id": "t1", "task_type": "convert", "task_status": s})
            for s in statuses
        ]
    )
    router.get(f"{URL}/v1/result/t1").respond(json=result)
    return submit


async def test_parse_success_with_pages() -> None:
    parser = DoclingParser(URL, poll_interval_s=0)
    with respx.mock(assert_all_called=True) as router:
        submit = mock_docling(router, recorded_result(), ["started", "success"])
        parsed = await parser.parse(FIXTURES / "sample.pdf", "sample.pdf", {"do_ocr": False})
    await parser.aclose()

    body = submit.calls.last.request.content.decode(errors="replace")
    assert 'name="to_formats"' in body and PAGE_BREAK in body
    assert 'name="do_ocr"\r\n\r\nfalse' in body
    assert PAGE_BREAK not in parsed.markdown
    assert parsed.pages is not None and len(parsed.pages) == 2
    assert parsed.page_at(parsed.markdown.index("Article 5")) == 2
    assert parsed.raw is not None and parsed.metadata["status"] == "success"


async def test_parse_failure_status() -> None:
    parser = DoclingParser(URL, poll_interval_s=0)
    with respx.mock() as router:
        router.post(f"{URL}/v1/convert/file/async").respond(
            json={"task_id": "t1", "task_type": "convert", "task_status": "pending"}
        )
        router.get(f"{URL}/v1/status/poll/t1").respond(
            json={"task_id": "t1", "task_type": "convert", "task_status": "failure",
                  "error_message": "broken pdf"}
        )  # fmt: skip
        with pytest.raises(ParserError, match="broken pdf"):
            await parser.parse(FIXTURES / "sample.pdf", "sample.pdf", {})
    await parser.aclose()


async def test_parse_timeout() -> None:
    parser = DoclingParser(URL, timeout_s=0.05, poll_interval_s=0.01)
    with respx.mock() as router:
        router.post(f"{URL}/v1/convert/file/async").respond(
            json={"task_id": "t1", "task_type": "convert", "task_status": "pending"}
        )
        router.get(f"{URL}/v1/status/poll/t1").respond(
            json={"task_id": "t1", "task_type": "convert", "task_status": "started"}
        )
        with pytest.raises(ParserError, match="timed out"):
            await parser.parse(FIXTURES / "sample.pdf", "sample.pdf", {})
    await parser.aclose()


async def test_unreachable_and_health() -> None:
    parser = DoclingParser(URL)
    with respx.mock() as router:
        router.post(f"{URL}/v1/convert/file/async").mock(side_effect=httpx.ConnectError("refused"))
        router.get(f"{URL}/health").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ParserUnavailableError):
            await parser.parse(FIXTURES / "sample.pdf", "sample.pdf", {})
        health = await parser.health()
    assert not health.available
    await parser.aclose()
