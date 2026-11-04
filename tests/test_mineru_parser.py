"""mineru-api HTTP client (the result zip is a recording of real MinerU 4.1.7 basic-tier output)."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from rag_service.parsers.base import ParserError, ParserUnavailableError
from rag_service.parsers.mineru import MinerUParser, page_spans_from_structured

FIXTURES = Path(__file__).parent / "fixtures"
URL = "http://mineru.unit"
TASK = "job_1"
FILE = "file-1"


def mock_mineru(
    router: respx.MockRouter,
    base: str = URL,
    statuses: list[str] | None = None,
    file_error: dict | None = None,
) -> dict[str, respx.Route]:
    statuses = statuses or ["pending", "running", "completed"]

    def task_detail(status: str) -> httpx.Response:
        f = {"file_id": FILE, "file_name": "sample.pdf",
             "status": "completed" if status == "completed" else ("failed" if status == "failed" else status)}  # fmt: skip
        if status == "completed":
            f["parse"] = {"tier": "basic", "duration_ms": 3840, "parser_version": "4.1.7"}
            f["result_url"] = f"{base}/v1/files/{FILE}/content"
        if file_error:
            f["error"] = file_error
        return httpx.Response(200, json={"task_id": TASK, "status": status, "files": [f]})

    routes = {
        "health": router.get(f"{base}/v1/health").respond(json={"status": "ok", "version": "4.1.7"}),
        "tiers": router.get(f"{base}/v1/tiers").respond(
            json={"object": "list", "data": [{"id": "flash"}, {"id": "basic"}]}
        ),
        "create": router.post(f"{base}/v1/tasks").respond(
            202,
            json={"task_id": TASK, "status": "waiting-file",
                  "files": [{"file_id": FILE, "file_name": "sample.pdf", "status": "waiting-file"}]},
        ),  # fmt: skip
        "upload": router.put(f"{base}/v1/tasks/{TASK}/uploads/{FILE}").respond(
            json={"task_id": TASK, "file_id": FILE, "status": "pending"}
        ),
        "poll": router.get(f"{base}/v1/tasks/{TASK}").mock(
            side_effect=[task_detail(s) for s in statuses] + [task_detail(statuses[-1])] * 1000
        ),
        "content": router.get(f"{base}/v1/files/{FILE}/content").respond(
            content=(FIXTURES / "mineru_result.zip").read_bytes(),
            headers={"content-type": "application/zip"},
        ),
        "cancel": router.post(f"{base}/v1/tasks/{TASK}/cancel").respond(
            json={"task_id": TASK, "status": "canceled"}
        ),
    }
    return routes


async def test_parse_success_with_pages() -> None:
    parser = MinerUParser(URL, poll_interval_s=0)
    opts = parser.validate_options({"ocr_mode": "txt"})
    assert opts == {"tier": "basic", "ocr_mode": "txt"}
    with respx.mock(assert_all_called=False) as router:
        routes = mock_mineru(router)
        parsed = await parser.parse(FIXTURES / "sample.pdf", "sample.pdf", opts)
    await parser.aclose()

    sent = json.loads(routes["create"].calls.last.request.content)
    assert sent["tier"] == "basic" and sent["ocr_mode"] == "txt"
    assert sent["files"][0]["bytes"] == (FIXTURES / "sample.pdf").stat().st_size
    assert routes["upload"].calls.last.request.content == (FIXTURES / "sample.pdf").read_bytes()

    assert "Article 5. Termination" in parsed.markdown
    assert parsed.pages is not None and [p.page for p in parsed.pages] == [1, 2]
    assert parsed.page_at(parsed.markdown.index("Either party")) == 2
    assert parsed.page_at(parsed.markdown.index("## Article 5")) == 2  # the heading marker is on page 2 too
    assert parsed.page_at(0) == 1
    assert parsed.metadata["backend"] == "tier:basic"
    assert parsed.raw is not None and len(parsed.raw["pages"]) == 2


async def test_parse_failure_reports_error() -> None:
    parser = MinerUParser(URL, poll_interval_s=0)
    with respx.mock(assert_all_called=False) as router:
        mock_mineru(router, statuses=["running", "failed"],
                    file_error={"code": "parse_failed", "message": "corrupted pdf"})  # fmt: skip
        with pytest.raises(ParserError, match="corrupted pdf"):
            await parser.parse(FIXTURES / "sample.pdf", "sample.pdf", {})
    await parser.aclose()


async def test_timeout_cancels_remote_task() -> None:
    parser = MinerUParser(URL, timeout_s=0.05, poll_interval_s=0.01)
    with respx.mock(assert_all_called=False) as router:
        routes = mock_mineru(router, statuses=["running"])
        with pytest.raises(ParserError, match="timed out"):
            await parser.parse(FIXTURES / "sample.pdf", "sample.pdf", {})
    assert routes["cancel"].called
    await parser.aclose()


async def test_unreachable_and_health() -> None:
    parser = MinerUParser(URL)
    with respx.mock() as router:
        router.post(f"{URL}/v1/tasks").mock(side_effect=httpx.ConnectError("refused"))
        router.get(f"{URL}/v1/health").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ParserUnavailableError):
            await parser.parse(FIXTURES / "sample.pdf", "sample.pdf", {})
        assert not (await parser.health()).available
    await parser.aclose()


def test_option_validation_and_tier_default() -> None:
    p = MinerUParser(URL, default_tier="flash")
    assert p.validate_options({}) == {"tier": "flash"}
    assert p.validate_options({"tier": "standard"})["tier"] == "standard"
    with pytest.raises(ValueError, match="tier"):
        p.validate_options({"tier": "ultra"})
    with pytest.raises(ValueError, match="unsupported"):
        p.validate_options({"backend": "pipeline"})
    with pytest.raises(ValueError):
        MinerUParser(URL, default_tier="gpu")


def test_page_spans_skips_unlocatable_pages() -> None:
    md = "# Title\n\nfirst page text\n\nthird page text"
    structured = {
        "pages": [
            {"page_idx": 0, "blocks": [{"type": "title", "content": "Title"}]},
            {"page_idx": 1, "blocks": [{"type": "image", "content": None}]},
            {"page_idx": 2, "blocks": [{"type": "text", "content": "third page text"}]},
        ]
    }
    spans = page_spans_from_structured(md, structured)
    assert spans is not None and [(s.page, s.start) for s in spans] == [(1, 0), (3, md.index("third"))]
    heading = page_spans_from_structured("p1\n\n## Next", {"pages": [
        {"page_idx": 0, "blocks": [{"content": "p1"}]},
        {"page_idx": 1, "blocks": [{"content": "Next"}]},
    ]})  # fmt: skip
    assert heading is not None and heading[1].start == len("p1\n\n")
    assert spans[-1].end == len(md)
    assert page_spans_from_structured(md, {"pages": []}) is None
