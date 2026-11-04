"""Datasets with a LiteLLM embedding model, against a mocked OpenAI-compatible /embeddings endpoint."""

import base64
import json
import struct
from typing import Any

import httpx
import respx

from rag_service.providers.embeddings import HashEmbedding
from rag_service.state import AppState
from tests.conftest import FIXTURES, LITELLM_TEST_URL, wait_for_job

KEY = "sk-test-embed-key-987654"
EMBED_URL = f"{LITELLM_TEST_URL}/embeddings"
_hash = HashEmbedding(model_name="mock", dim=16)


def fake_embeddings(dim: int | None = None) -> Any:
    """respx side effect imitating LiteLLM's /embeddings (lexical vectors, so ranking is meaningful)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        data = []
        for i, text in enumerate(inputs):
            vec = _hash.get_text_embedding(text)
            if dim is not None:
                vec = (vec * (dim // len(vec) + 1))[:dim]
            if body.get("encoding_format") == "base64":
                emb: Any = base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode()
            else:
                emb = vec
            data.append({"object": "embedding", "index": i, "embedding": emb})
        return httpx.Response(
            200,
            json={"object": "list", "data": data, "model": body["model"],
                  "usage": {"prompt_tokens": 5, "total_tokens": 5}},
        )  # fmt: skip

    return handler


def sent(route: respx.Route) -> list[dict]:
    return [json.loads(c.request.content) for c in route.calls]


async def create_litellm_dataset(client: httpx.AsyncClient, **embedding: Any) -> httpx.Response:
    return await client.post(
        "/datasets",
        headers={"X-Model-API-Key": KEY},
        json={"name": "bedrock", "embedding": {"provider": "litellm", **embedding}, "chunk_size": 256},
    )


async def test_create_probes_dimension(client: httpx.AsyncClient, state: AppState) -> None:
    with respx.mock() as router:
        route = router.post(EMBED_URL).mock(side_effect=fake_embeddings())
        r = await create_litellm_dataset(client, model="titan-embed-v2", dimensions=16)
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["embedding"] == {"provider": "litellm", "model": "titan-embed-v2", "dim": 16,
                               "dimensions": 16, "input_type": None}  # fmt: skip
    assert route.call_count == 1
    req = route.calls.last.request
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert sent(route)[0]["model"] == "titan-embed-v2" and sent(route)[0]["dimensions"] == 16
    await client.delete(f"/datasets/{ds['id']}")


async def test_create_validation_and_upstream_errors(client: httpx.AsyncClient, state: AppState) -> None:
    r = await client.post(
        "/datasets", json={"name": "x", "embedding": {"provider": "litellm", "model": "titan-embed-v2"}}
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "missing_model_api_key"

    r = await create_litellm_dataset(client)
    assert r.status_code == 422 and r.json()["error"]["code"] == "embedding_model_required"

    with respx.mock() as router:
        router.post(EMBED_URL).respond(
            400, json={"error": {"message": "Invalid model name passed in model=titan-embed-v9"}}
        )
        r = await create_litellm_dataset(client, model="titan-embed-v9")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_embedding_model"
    assert "titan-embed-v9" in r.json()["error"]["message"]

    with respx.mock() as router:
        router.post(EMBED_URL).mock(side_effect=fake_embeddings(dim=3072))
        r = await create_litellm_dataset(client, model="big-embed")
    assert r.status_code == 422 and r.json()["error"]["code"] == "embedding_dim_too_large"

    with respx.mock() as router:
        router.post(EMBED_URL).respond(401, json={"error": {"message": f"bad key {KEY}"}})
        r = await create_litellm_dataset(client, model="titan-embed-v2")
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_auth_failed"
    assert KEY not in r.text

    original = state.settings.litellm_base_url
    state.settings.litellm_base_url = None
    try:
        r = await create_litellm_dataset(client, model="titan-embed-v2")
    finally:
        state.settings.litellm_base_url = original
    assert r.status_code == 503 and r.json()["error"]["code"] == "litellm_not_configured"

    # nothing was created by the failed attempts
    r = await client.get("/datasets")
    assert all(d["name"] != "bedrock" for d in r.json()["items"])


async def test_index_and_search_send_input_types(client: httpx.AsyncClient) -> None:
    input_type = {"document": "search_document", "query": "search_query"}
    with respx.mock() as router:
        route = router.post(EMBED_URL).mock(side_effect=fake_embeddings())
        r = await create_litellm_dataset(client, model="cohere-embed-v4", input_type=input_type)
        assert r.status_code == 201, r.text
        ds_id = r.json()["id"]
        assert r.json()["embedding"]["input_type"] == input_type

        r = await client.post(
            f"/datasets/{ds_id}/documents",
            headers={"X-Model-API-Key": KEY},
            files={"file": ("contract.md", (FIXTURES / "contract.md").read_bytes())},
            data={"index": "true"},
        )
        assert r.status_code == 201, r.text
        job = await wait_for_job(client, ds_id, r.json()["job"]["id"])
        assert job["status"] == "succeeded", job

        r = await client.post(
            f"/datasets/{ds_id}/search",
            headers={"X-Model-API-Key": KEY},
            json={"query": "계약 해지 조건", "top_k": 1},
        )
        assert r.status_code == 200, r.text
        assert r.json()["results"][0]["section_path"][-1] == "제5조 계약의 해지"

        bodies = sent(route)
        assert {b["input_type"] for b in bodies[:-1]} == {"search_document"}  # probe + indexing
        assert bodies[-1]["input_type"] == "search_query" and bodies[-1]["input"] == ["계약 해지 조건"]
        assert {b["model"] for b in bodies} == {"cohere-embed-v4"}

    r = await client.post(f"/datasets/{ds_id}/search", json={"query": "x"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "missing_model_api_key"
    await client.delete(f"/datasets/{ds_id}")


async def test_indexing_upstream_failure_fails_job_without_leaking_key(client: httpx.AsyncClient) -> None:
    with respx.mock() as router:
        router.post(EMBED_URL).mock(side_effect=fake_embeddings())
        r = await create_litellm_dataset(client, model="titan-embed-v2")
        ds_id = r.json()["id"]

    with respx.mock() as router:
        router.post(EMBED_URL).respond(401, json={"error": {"message": f"Invalid key {KEY}"}})
        r = await client.post(
            f"/datasets/{ds_id}/documents",
            headers={"X-Model-API-Key": KEY},
            files={"file": ("notes.txt", (FIXTURES / "notes.txt").read_bytes())},
            data={"index": "true"},
        )
        job = await wait_for_job(client, ds_id, r.json()["job"]["id"])
    assert job["status"] == "failed" and job["stage"] == "embedding"
    assert job["error"].startswith("embedding failed (upstream_auth_failed)")
    assert KEY not in json.dumps(job)
    doc = (await client.get(f"/datasets/{ds_id}/documents/{job['document_id']}")).json()
    assert doc["status"] == "failed"
    await client.delete(f"/datasets/{ds_id}")
