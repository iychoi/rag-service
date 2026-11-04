"""/query: retrieval + LLM answer through a mocked LiteLLM (OpenAI-compatible) endpoint."""

import json

import httpx
import respx

from rag_service.models.schemas import SearchResult
from rag_service.services.answering import build_user_prompt, extract_citations, number_sources
from rag_service.state import AppState
from tests.conftest import FIXTURES, LITELLM_TEST_URL, wait_for_job

KEY = "sk-test-key-123456"
CHAT_URL = f"{LITELLM_TEST_URL}/chat/completions"


def chat_completion(content: str, model: str = "claude-sonnet-5-5") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1793823516,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 412, "completion_tokens": 23, "total_tokens": 435},
    }


async def indexed_dataset(client: httpx.AsyncClient, dataset: dict) -> str:
    ds_id = dataset["id"]
    r = await client.post(
        f"/datasets/{ds_id}/documents",
        files={"file": ("contract.md", (FIXTURES / "contract.md").read_bytes())},
        data={"index": "true", "metadata": json.dumps({"year": 2025})},
    )
    assert r.status_code == 201, r.text
    assert (await wait_for_job(client, ds_id, r.json()["job"]["id"]))["status"] == "succeeded"
    return ds_id


# ---------------- unit ----------------


def _result(text: str, **kw: object) -> SearchResult:
    base = dict(score=0.5, text=text, document_id="d" * 32, filename="a.pdf", page=None,
                section_path=[], chunk_index=0, metadata={})  # fmt: skip
    return SearchResult(**{**base, **kw})


def test_prompt_and_citations() -> None:
    sources = number_sources(
        [_result("alpha", page=2, section_path=["A", "B"]), _result("beta", filename=None)]
    )
    assert [s.id for s in sources] == [1, 2]
    prompt = build_user_prompt("what?", sources)
    assert "[1] (a.pdf, page 2, A > B)\nalpha" in prompt
    assert "[2] (unknown file)\nbeta" in prompt
    assert prompt.endswith("Question: what?")
    assert extract_citations("x [2] y [1][2] z [7] [0]", 2) == [2, 1]
    assert extract_citations("no refs", 3) == []


# ---------------- API ----------------


async def test_query_answers_with_citations(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = await indexed_dataset(client, dataset)
    with respx.mock(assert_all_called=True) as router:
        route = router.post(CHAT_URL).respond(
            json=chat_completion("30일 전에 서면으로 통지하면 해지할 수 있습니다 [1]. 연체 시 회사도 해지 가능 [1][3].")
        )
        r = await client.post(
            f"/datasets/{ds_id}/query",
            headers={"X-Model-API-Key": KEY},
            json={"query": "계약 해지 조건은?", "top_k": 3, "filters": {"metadata": {"year": 2025}}},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].startswith("30일 전에")
    assert body["model"] == "claude-sonnet-5-5"
    assert body["citations"] == [1, 3]
    assert [s["id"] for s in body["sources"]] == [1, 2, 3]
    assert body["sources"][0]["section_path"] == ["클라우드 서비스 이용 약관", "제5조 계약의 해지"]
    assert body["usage"] == {"input_tokens": 412, "output_tokens": 23}

    sent = route.calls.last.request
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    payload = json.loads(sent.content)
    assert payload["model"] == "claude-sonnet-5-5"
    assert payload["messages"][0]["role"] == "system"
    user = payload["messages"][1]["content"]
    assert "[1] (contract.md, 클라우드 서비스 이용 약관 > 제5조 계약의 해지)" in user
    assert user.endswith("Question: 계약 해지 조건은?")


async def test_query_model_selection(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = await indexed_dataset(client, dataset)
    r = await client.post(
        f"/datasets/{ds_id}/query",
        headers={"X-Model-API-Key": KEY},
        json={"query": "해지", "model": "gpt-6-luna"},
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "model_not_allowed"

    with respx.mock() as router:
        route = router.post(CHAT_URL).respond(json=chat_completion("ok [1]", "claude-haiku-5-5"))
        r = await client.post(
            f"/datasets/{ds_id}/query",
            headers={"X-Model-API-Key": KEY},
            json={"query": "해지", "model": "claude-haiku-5-5", "max_output_tokens": 200},
        )
    assert r.status_code == 200 and r.json()["model"] == "claude-haiku-5-5"
    payload = json.loads(route.calls.last.request.content)
    assert payload["model"] == "claude-haiku-5-5" and payload["max_tokens"] == 200


async def test_query_requires_key(client: httpx.AsyncClient, dataset: dict) -> None:
    r = await client.post(f"/datasets/{dataset['id']}/query", json={"query": "해지"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "missing_model_api_key"


async def test_query_without_sources_skips_llm(client: httpx.AsyncClient, dataset: dict) -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.post(CHAT_URL).respond(json=chat_completion("should not be called"))
        r = await client.post(
            f"/datasets/{dataset['id']}/query", headers={"X-Model-API-Key": KEY}, json={"query": "x"}
        )
    assert r.status_code == 200
    assert r.json() == {"query": "x", "answer": None, "model": None, "citations": [],
                        "sources": [], "usage": None}  # fmt: skip
    assert not route.called


async def test_query_upstream_errors(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = await indexed_dataset(client, dataset)
    url = f"/datasets/{ds_id}/query"
    headers = {"X-Model-API-Key": KEY}

    with respx.mock() as router:
        router.post(CHAT_URL).respond(
            401, json={"error": {"message": f"Invalid proxy server token passed. key={KEY}"}}
        )
        r = await client.post(url, headers=headers, json={"query": "해지"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_auth_failed"
    assert KEY not in r.text

    with respx.mock() as router:
        router.post(CHAT_URL).respond(
            400, json={"error": {"message": f"Invalid model name passed in model=x, key {KEY}"}}
        )
        r = await client.post(url, headers=headers, json={"query": "해지"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"
    assert "HTTP 400" in r.json()["error"]["message"] and KEY not in r.text

    with respx.mock() as router:
        router.post(CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))
        r = await client.post(url, headers=headers, json={"query": "해지"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_unreachable"

    with respx.mock() as router:
        router.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        r = await client.post(url, headers=headers, json={"query": "해지"})
    assert r.status_code == 504 and r.json()["error"]["code"] == "upstream_timeout"


async def test_query_not_configured(client: httpx.AsyncClient, dataset: dict, state: AppState) -> None:
    original = state.settings.litellm_base_url
    state.settings.litellm_base_url = None
    try:
        r = await client.post(
            f"/datasets/{dataset['id']}/query", headers={"X-Model-API-Key": KEY}, json={"query": "x"}
        )
    finally:
        state.settings.litellm_base_url = original
    assert r.status_code == 503 and r.json()["error"]["code"] == "litellm_not_configured"
