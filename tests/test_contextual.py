"""Contextual retrieval: per-chunk contexts from a mocked LiteLLM chat endpoint."""

import json
from typing import Any

import httpx
import respx

from rag_service.services.contextual import ContextGenerator, split_into_parts
from rag_service.services.tokens import count_tokens
from tests.conftest import FIXTURES, LITELLM_TEST_URL, wait_for_job

KEY = "sk-test-context-key-24680"
CHAT_URL = f"{LITELLM_TEST_URL}/chat/completions"
MODEL = "claude-haiku-5-5"
TERMINATION_CONTEXT = "Termination clause of the cloud storage terms: how either side may cancel the agreement."


def completion(content: str, cached: int = 0) -> dict:
    return {
        "id": "c1", "object": "chat.completion", "created": 1, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }  # fmt: skip


def chunk_of(body: dict) -> str:
    return body["messages"][0]["content"][1]["text"].split("<chunk>\n", 1)[1].split("\n</chunk>")[0]


def fake_chat(calls: list[dict]) -> Any:
    """Probe calls (max_tokens=1) get "OK"; context calls get a context naming the chunk's clause."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if body.get("max_tokens") == 1:
            return httpx.Response(200, json=completion("OK"))
        chunk = chunk_of(body)
        text = TERMINATION_CONTEXT if "해지" in chunk else f"Part of the cloud storage terms ({len(chunk)} chars)."
        return httpx.Response(200, json=completion(text, cached=80 if len(calls) > 2 else 0))

    return handler


async def create(client: httpx.AsyncClient, key: str | None = KEY, model: str = MODEL) -> httpx.Response:
    return await client.post(
        "/datasets",
        headers={"X-Model-API-Key": key} if key else {},
        json={"name": "ctx", "embedding": {"provider": "hash"}, "chunk_size": 256,
              "contextual": {"model": model}},
    )  # fmt: skip


# ---------------- unit ----------------


def test_split_into_parts() -> None:
    chunks = ["alpha " * 50, "beta " * 50, "gamma " * 50]
    assert split_into_parts(chunks, 10_000) == [[0, 1, 2]]
    assert split_into_parts(chunks, 120) == [[0, 1], [2]]
    assert split_into_parts(chunks, 1) == [[0], [1], [2]]  # a part always has one chunk
    assert split_into_parts([], 100) == []


async def test_generator_caches_document_and_keeps_order() -> None:
    calls: list[dict] = []
    chunks = [f"clause {i} " * 10 for i in range(5)]
    two_chunks = 2 * (count_tokens(chunks[0]) + 1)  # → parts of 2, 2, 1 chunks
    gen = ContextGenerator(base_url=LITELLM_TEST_URL, model=MODEL, api_key=KEY,
                           max_output_tokens=150, document_tokens=two_chunks, concurrency=3,
                           timeout_s=10, max_retries=0)  # fmt: skip
    progress: list[int] = []

    async def on_progress(done: int) -> None:
        progress.append(done)

    with respx.mock() as router:
        router.post(CHAT_URL).mock(side_effect=fake_chat(calls))
        contexts = await gen.generate(chunks, on_progress)

    assert contexts == [f"Part of the cloud storage terms ({len(c)} chars)." for c in chunks]
    assert sorted(progress) == [1, 2, 3, 4, 5]
    assert gen.usage.calls == 5 and gen.usage.output_tokens == 60 and gen.usage.cached_input_tokens > 0

    first = calls[0]
    assert first["model"] == MODEL and first["max_tokens"] == 150 and first["temperature"] == 0.0
    doc_block = first["messages"][0]["content"][0]
    assert doc_block["cache_control"] == {"type": "ephemeral"}
    assert doc_block["text"].startswith("<document>\n(Part 1 of 3 of the document)\n\nclause 0")
    # Each part's first chunk is sent before the others of that part (it writes the cache)
    order = [chunk_of(c) for c in calls]
    assert order[0] == chunks[0] and order[2] == chunks[2] and order[4] == chunks[4]


# ---------------- API ----------------


async def test_dataset_creation_checks_the_model(client: httpx.AsyncClient) -> None:
    r = await create(client, key=None)
    assert r.status_code == 400 and r.json()["error"]["code"] == "missing_model_api_key"

    with respx.mock() as router:
        router.post(CHAT_URL).respond(
            400, json={"error": {"message": "Invalid model name passed in model=claude-haiku-9"}}
        )
        r = await create(client, model="claude-haiku-9")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_contextual_model"
    assert "claude-haiku-9" in r.json()["error"]["message"]

    calls: list[dict] = []
    with respx.mock() as router:
        router.post(CHAT_URL).mock(side_effect=fake_chat(calls))
        r = await create(client)
    assert r.status_code == 201, r.text
    assert r.json()["contextual"] == {"model": MODEL}
    assert len(calls) == 1 and calls[0]["max_tokens"] == 1
    await client.delete(f"/datasets/{r.json()['id']}")


async def test_indexing_embeds_contexts(client: httpx.AsyncClient) -> None:
    calls: list[dict] = []
    with respx.mock() as router:
        router.post(CHAT_URL).mock(side_effect=fake_chat(calls))
        ds = (await create(client)).json()
        ds_id = ds["id"]
        try:
            r = await client.post(
                f"/datasets/{ds_id}/documents",
                files={"file": ("contract.md", (FIXTURES / "contract.md").read_bytes())},
                data={"index": "true"},
            )
            assert r.status_code == 400  # the key is needed for the contexts even with hash embeddings

            r = await client.post(
                f"/datasets/{ds_id}/documents",
                headers={"X-Model-API-Key": KEY},
                files={"file": ("contract.md", (FIXTURES / "contract.md").read_bytes())},
                data={"index": "true"},
            )
            assert r.status_code == 201, r.text
            job = await wait_for_job(client, ds_id, r.json()["job"]["id"])
            assert job["status"] == "succeeded", job

            context_calls = [c for c in calls if c.get("max_tokens") != 1]
            doc = (await client.get(f"/datasets/{ds_id}/documents/{r.json()['document']['id']}")).json()
            assert len(context_calls) == doc["num_chunks"]
            sent_doc = context_calls[0]["messages"][0]["content"][0]["text"]
            assert "제5조 계약의 해지" in sent_doc and "제7조 손해배상" in sent_doc  # the whole document

            # The termination chunk is found by words that only its context contains
            r = await client.post(f"/datasets/{ds_id}/search",
                                  json={"query": "cancel the agreement", "top_k": 1})  # fmt: skip
            top = r.json()["results"][0]
            assert top["section_path"][-1] == "제5조 계약의 해지"
            assert top["context"] == TERMINATION_CONTEXT
            assert "cancel" not in top["text"]  # the stored text is the chunk itself
        finally:
            await client.delete(f"/datasets/{ds_id}")


async def test_context_failure_fails_the_job(client: httpx.AsyncClient) -> None:
    calls: list[dict] = []
    with respx.mock() as router:
        router.post(CHAT_URL).mock(side_effect=fake_chat(calls))
        ds_id = (await create(client)).json()["id"]
    try:
        with respx.mock() as router:
            router.post(CHAT_URL).respond(401, json={"error": {"message": f"bad key {KEY}"}})
            r = await client.post(
                f"/datasets/{ds_id}/documents",
                headers={"X-Model-API-Key": KEY},
                files={"file": ("contract.md", (FIXTURES / "contract.md").read_bytes())},
                data={"index": "true"},
            )
            job = await wait_for_job(client, ds_id, r.json()["job"]["id"])
        assert job["status"] == "failed" and job["stage"] == "contextualizing"
        assert job["error"].startswith("context generation failed (upstream_auth_failed)")
        assert KEY not in job["error"]
        doc = (await client.get(f"/datasets/{ds_id}/documents/{r.json()['document']['id']}")).json()
        assert doc["status"] == "failed" and doc["num_chunks"] is None
    finally:
        await client.delete(f"/datasets/{ds_id}")
