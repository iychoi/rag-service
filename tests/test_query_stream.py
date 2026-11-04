"""/query with stream=true: Server-Sent Events over a mocked streaming LiteLLM endpoint."""

import json

import httpx
import respx

from tests.test_query import CHAT_URL, KEY, indexed_dataset

HEADERS = {"X-Model-API-Key": KEY}


def openai_stream(pieces: list[str], *, usage: bool = True, error_after: int | None = None) -> str:
    """Body of an OpenAI-compatible streaming chat completion (as LiteLLM sends it)."""

    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}  # fmt: skip

    events = [chunk({"role": "assistant", "content": ""})]
    for i, piece in enumerate(pieces):
        if error_after is not None and i == error_after:
            events.append({"error": {"message": f"upstream overloaded (key {KEY})"}})
            break
        events.append(chunk({"content": piece}))
    else:
        events.append(chunk({}, "stop"))
        if usage:
            events.append({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                           "choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 12,
                                                    "total_tokens": 312}})  # fmt: skip
    body = "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events)
    return body + ("data: [DONE]\n\n" if error_after is None else "")


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def mock_stream(router: respx.MockRouter, body: str) -> respx.Route:
    return router.post(CHAT_URL).respond(
        200, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


async def test_stream_events(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = await indexed_dataset(client, dataset)
    pieces = ["30일 전에 ", "서면으로 통지하면 ", "해지할 수 있습니다 [1]."]
    with respx.mock() as router:
        route = mock_stream(router, openai_stream(pieces))
        r = await client.post(
            f"/datasets/{ds_id}/query",
            headers=HEADERS,
            json={"query": "계약 해지 조건은?", "top_k": 3, "stream": True},
        )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"

    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names == ["sources", "delta", "delta", "delta", "done"]

    sources = events[0][1]
    assert sources["query"] == "계약 해지 조건은?"
    assert [s["id"] for s in sources["sources"]] == [1, 2, 3]

    deltas = "".join(d["text"] for e, d in events if e == "delta")
    assert deltas == "".join(pieces)
    done = events[-1][1]
    assert done == {
        "answer": "30일 전에 서면으로 통지하면 해지할 수 있습니다 [1].",
        "model": "claude-sonnet-5-5",
        "citations": [1],
        "usage": {"input_tokens": 300, "output_tokens": 12},
    }

    payload = json.loads(route.calls.last.request.content)
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


async def test_stream_upstream_error_before_start_is_json(
    client: httpx.AsyncClient, dataset: dict
) -> None:
    ds_id = await indexed_dataset(client, dataset)
    with respx.mock() as router:
        router.post(CHAT_URL).respond(401, json={"error": {"message": f"bad key {KEY}"}})
        r = await client.post(
            f"/datasets/{ds_id}/query", headers=HEADERS, json={"query": "해지", "stream": True}
        )
    assert r.status_code == 502
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["error"]["code"] == "upstream_auth_failed"
    assert KEY not in r.text


async def test_stream_error_mid_stream_is_an_event(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = await indexed_dataset(client, dataset)
    with respx.mock() as router:
        mock_stream(router, openai_stream(["first part ", "second part"], error_after=1))
        r = await client.post(
            f"/datasets/{ds_id}/query", headers=HEADERS, json={"query": "해지", "stream": True}
        )
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert [e for e, _ in events] == ["sources", "delta", "error"]
    assert events[1][1] == {"text": "first part "}
    err = events[-1][1]["error"]
    assert err["code"] == "upstream_error" and "overloaded" in err["message"]
    assert KEY not in r.text


async def test_stream_without_sources_skips_llm(client: httpx.AsyncClient, dataset: dict) -> None:
    with respx.mock(assert_all_called=False) as router:
        route = mock_stream(router, openai_stream(["never"]))
        r = await client.post(
            f"/datasets/{dataset['id']}/query", headers=HEADERS, json={"query": "x", "stream": True}
        )
    assert r.status_code == 200 and not route.called
    assert parse_sse(r.text) == [
        ("sources", {"query": "x", "sources": []}),
        ("done", {"answer": None, "model": None, "citations": [], "usage": None}),
    ]


async def test_stream_validation_errors_are_json(client: httpx.AsyncClient, dataset: dict) -> None:
    r = await client.post(
        f"/datasets/{dataset['id']}/query", json={"query": "x", "stream": True}
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "missing_model_api_key"
