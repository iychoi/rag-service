"""API end-to-end tests: real PostgreSQL (pgvector, testcontainers) + hash embedding + passthrough / mocked parsers."""

import asyncio
import json

import httpx
import respx
from sqlalchemy import text

from rag_service.parsers.docling import PAGE_BREAK
from rag_service.state import AppState
from tests.conftest import DOCLING_TEST_URL, FIXTURES, MINERU_TEST_URL, wait_for_job
from tests.test_mineru_parser import mock_mineru


async def table_exists(state: AppState, ds_id: str) -> bool:
    async with state.db.async_engine.connect() as conn:
        res = await conn.execute(
            text("SELECT to_regclass(:t) IS NOT NULL"), {"t": f"vectors.data_ds_{ds_id}"}
        )
        return bool(res.scalar_one())


async def upload(
    client: httpx.AsyncClient, ds_id: str, name: str, data: bytes, **form: str
) -> httpx.Response:
    return await client.post(
        f"/datasets/{ds_id}/documents", files={"file": (name, data)}, data=form
    )


# ---------------- datasets ----------------


async def test_dataset_lifecycle(client: httpx.AsyncClient, state: AppState) -> None:
    r = await client.post("/datasets", json={"name": "x"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = await client.post("/datasets", json={"name": "x", "embedding": {"provider": "openai"}})
    assert r.status_code == 422
    r = await client.post(
        "/datasets", json={"name": "x", "embedding": {"provider": "hash"}, "chunk_size": 9000}
    )
    assert r.status_code == 422
    r = await client.post(
        "/datasets",
        json={"name": "x", "embedding": {"provider": "hash"}, "chunk_size": 128, "chunk_overlap": 128},
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_chunk_overlap"

    r = await client.post(
        "/datasets",
        json={"name": "legal", "embedding": {"provider": "hash", "dimensions": 256},
              "external_ref": "ext-1", "metadata": {"team": "legal"}},
    )  # fmt: skip
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["embedding"] == {"provider": "hash", "model": "hash-ngram-v1", "dim": 256,
                               "dimensions": 256, "input_type": None}  # fmt: skip
    assert ds["metadata"] == {"team": "legal"}
    assert await table_exists(state, ds["id"])

    r = await client.post(
        "/datasets", json={"name": "dup", "embedding": {"provider": "hash"}, "external_ref": "ext-1"}
    )
    assert r.status_code == 409

    r = await client.get("/datasets", params={"external_ref": "ext-1"})
    assert [d["id"] for d in r.json()["items"]] == [ds["id"]]
    r = await client.get("/datasets", params={"ids": f"{ds['id']},{'0' * 32}"})
    assert r.json()["total"] == 1

    r = await client.patch(f"/datasets/{ds['id']}", json={"name": "legal2", "description": "d"})
    assert r.json()["name"] == "legal2" and r.json()["description"] == "d"
    r = await client.patch(f"/datasets/{ds['id']}", json={"embedding": {"provider": "hash"}})
    assert r.status_code == 422

    r = await client.delete(f"/datasets/{ds['id']}")
    assert r.status_code == 204
    assert not await table_exists(state, ds["id"])
    assert (await client.get(f"/datasets/{ds['id']}")).status_code == 404
    assert (await client.get("/datasets/not-a-valid-id")).status_code == 404


# ---------------- documents + indexing + search ----------------


async def test_upload_index_search_flow(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = dataset["id"]
    contract = (FIXTURES / "contract.md").read_bytes()
    r = await upload(
        client, ds_id, "contract.md", contract, index="true", metadata=json.dumps({"year": 2025})
    )
    assert r.status_code == 201, r.text
    body = r.json()
    doc, job = body["document"], body["job"]
    assert job["parser"] == "passthrough" and job["status"] == "queued"

    job = await wait_for_job(client, ds_id, job["id"])
    assert job["status"] == "succeeded", job
    assert job["progress"] == 1.0

    r = await upload(client, ds_id, "notes.txt", (FIXTURES / "notes.txt").read_bytes(), index="true")
    notes_job = await wait_for_job(client, ds_id, r.json()["job"]["id"])
    assert notes_job["status"] == "succeeded"
    notes_id = r.json()["document"]["id"]

    r = await client.get(f"/datasets/{ds_id}/documents/{doc['id']}")
    d = r.json()
    assert d["status"] == "indexed" and d["parser"] == "passthrough" and d["num_chunks"] >= 4
    assert d["last_job"]["status"] == "succeeded"

    r = await client.post(f"/datasets/{ds_id}/search", json={"query": "계약 해지 조건", "top_k": 3})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert results[0]["document_id"] == doc["id"]
    assert results[0]["section_path"] == ["클라우드 서비스 이용 약관", "제5조 계약의 해지"]
    assert results[0]["filename"] == "contract.md"
    assert results[0]["metadata"] == {"year": 2025}

    # document filter / metadata filter (types must match exactly)
    r = await client.post(
        f"/datasets/{ds_id}/search",
        json={"query": "storage", "top_k": 5, "filters": {"document_ids": [notes_id]}},
    )
    assert {x["document_id"] for x in r.json()["results"]} == {notes_id}
    r = await client.post(
        f"/datasets/{ds_id}/search",
        json={"query": "해지", "top_k": 5, "filters": {"metadata": {"year": 2025}}},
    )
    assert r.json()["results"] and {x["document_id"] for x in r.json()["results"]} == {doc["id"]}
    r = await client.post(
        f"/datasets/{ds_id}/search",
        json={"query": "해지", "top_k": 5, "filters": {"metadata": {"year": "2025"}}},
    )
    assert r.json()["results"] == []

    # metadata changes are reflected in search filters immediately
    r = await client.patch(
        f"/datasets/{ds_id}/documents/{doc['id']}", json={"metadata": {"year": 2026}}
    )
    assert r.status_code == 200 and r.json()["metadata"] == {"year": 2026}
    r = await client.post(
        f"/datasets/{ds_id}/search",
        json={"query": "해지", "top_k": 5, "filters": {"metadata": {"year": 2026}}},
    )
    assert {x["document_id"] for x in r.json()["results"]} == {doc["id"]}

    # parsed output and original download
    r = await client.get(f"/datasets/{ds_id}/documents/{doc['id']}/parsed")
    assert r.status_code == 200 and "제5조" in r.text
    r = await client.get(f"/datasets/{ds_id}/documents/{doc['id']}/content")
    assert r.content == contract

    # duplicate upload
    r = await upload(client, ds_id, "copy.md", contract)
    assert r.status_code == 409 and doc["id"] in r.json()["error"]["message"]

    # invalid filters/requests
    r = await client.post(f"/datasets/{ds_id}/search", json={"query": "x", "top_k": 500})
    assert r.status_code == 422
    r = await client.post(
        f"/datasets/{ds_id}/search", json={"query": "x", "filters": {"document_ids": ["' OR 1=1"]}}
    )
    assert r.status_code == 422
    r = await client.post(
        f"/datasets/{ds_id}/search",
        json={"query": "x", "filters": {"metadata": {"a'; DROP TABLE x;--": "v"}}},
    )
    assert r.status_code == 422
    # quotes in values are safe because values are bound (no results)
    r = await client.post(
        f"/datasets/{ds_id}/search",
        json={"query": "x", "filters": {"metadata": {"year": "' OR '1'='1"}}},
    )
    assert r.status_code == 200 and r.json()["results"] == []


async def test_reindex_replaces_vectors(
    client: httpx.AsyncClient, dataset: dict, state: AppState
) -> None:
    ds_id = dataset["id"]
    r = await upload(client, ds_id, "c.md", (FIXTURES / "contract.md").read_bytes(), index="true")
    doc_id = r.json()["document"]["id"]
    first = await wait_for_job(client, ds_id, r.json()["job"]["id"])
    assert first["status"] == "succeeded"

    from rag_service.models.tables import Dataset

    async with state.db.sessionmaker() as s:
        ds = await s.get(Dataset, ds_id)
    n1 = await state.vectors.count_document_nodes(ds, doc_id)

    r = await client.post(
        f"/datasets/{ds_id}/documents/{doc_id}/index", json={"parser": "docling", "force_reparse": True}
    )
    assert r.status_code == 202, r.text
    second = await wait_for_job(client, ds_id, r.json()["id"])
    assert second["status"] == "succeeded"
    assert await state.vectors.count_document_nodes(ds, doc_id) == n1  # replaced, not accumulated

    r = await client.get(f"/datasets/{ds_id}/jobs", params={"document_id": doc_id})
    assert r.json()["total"] == 2

    r = await client.delete(f"/datasets/{ds_id}/documents/{doc_id}")
    assert r.status_code == 204
    assert await state.vectors.count_document_nodes(ds, doc_id) == 0
    assert (await client.get(f"/datasets/{ds_id}/documents/{doc_id}")).status_code == 404


async def test_dataset_boundary(client: httpx.AsyncClient, dataset: dict) -> None:
    """Documents/jobs are not reachable through another dataset's path."""
    r = await upload(client, dataset["id"], "n.txt", b"boundary test", index="true")
    doc_id, job_id = r.json()["document"]["id"], r.json()["job"]["id"]
    await wait_for_job(client, dataset["id"], job_id)

    other = (
        await client.post("/datasets", json={"name": "other", "embedding": {"provider": "hash"}})
    ).json()
    try:
        for path in (
            f"/datasets/{other['id']}/documents/{doc_id}",
            f"/datasets/{other['id']}/documents/{doc_id}/content",
            f"/datasets/{other['id']}/jobs/{job_id}",
        ):
            assert (await client.get(path)).status_code == 404, path
        assert (await client.delete(f"/datasets/{other['id']}/documents/{doc_id}")).status_code == 404
        r = await client.post(f"/datasets/{other['id']}/search", json={"query": "boundary"})
        assert r.json()["results"] == []
    finally:
        await client.delete(f"/datasets/{other['id']}")


async def test_upload_validation(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = dataset["id"]
    r = await upload(client, ds_id, "a.exe", b"MZ", index="true")
    assert r.status_code == 422 and r.json()["error"]["code"] == "unsupported_parser_or_format"
    r = await upload(client, ds_id, "a.pdf", b"%PDF", index="true", parser="nope")
    assert r.status_code == 422
    r = await upload(client, ds_id, "a.pdf", b"%PDF", index="true", parser="mineru",
                     options='{"tier": "ultra"}')  # fmt: skip
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_parser_options"
    r = await upload(client, ds_id, "a.pdf", b"%PDF", index="true", parser="mineru")
    assert r.status_code == 503  # no mineru service
    r = await upload(client, ds_id, "a.md", b"x", metadata="[1]")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_json"
    r = await upload(client, ds_id, "a.md", b"x", metadata=json.dumps({"job_id": "x"}))
    assert r.status_code == 422 and "reserved" in r.json()["error"]["message"]
    # without a docling service, indexing requests get 503 but plain uploads succeed
    r = await upload(client, ds_id, "a.pdf", b"%PDF-1.4", index="true")
    assert r.status_code == 503 and r.json()["error"]["code"] == "parser_unavailable"
    r = await upload(client, ds_id, "a.pdf", b"%PDF-1.4")
    assert r.status_code == 201 and r.json()["job"] is None
    assert r.json()["document"]["status"] == "uploaded"

    r = await client.get("/parsers")
    by_name = {p["name"]: p for p in r.json()}
    assert by_name["passthrough"]["available"] and not by_name["docling"]["available"]
    assert not by_name["mineru"]["available"]


# ---------------- docling (mocked service) ----------------


def docling_routes(router: respx.MockRouter, statuses: list[str] | None = None) -> None:
    data = json.loads((FIXTURES / "docling_result.json").read_text())
    data["document"]["md_content"] = data["document"]["md_content"].replace(
        "<!-- page-break -->", PAGE_BREAK
    )
    router.get(f"{DOCLING_TEST_URL}/health").respond(json={"status": "ok"})
    router.get(f"{DOCLING_TEST_URL}/version").respond(json={"docling-serve": "1.38.2"})
    router.post(f"{DOCLING_TEST_URL}/v1/convert/file/async").respond(
        json={"task_id": "t1", "task_type": "convert", "task_status": "pending"}
    )
    if statuses is None:
        router.get(f"{DOCLING_TEST_URL}/v1/status/poll/t1").respond(
            json={"task_id": "t1", "task_type": "convert", "task_status": "success"}
        )
    else:
        router.get(f"{DOCLING_TEST_URL}/v1/status/poll/t1").respond(
            json={"task_id": "t1", "task_type": "convert", "task_status": statuses[0]}
        )
    router.get(f"{DOCLING_TEST_URL}/v1/result/t1").respond(json=data)


async def test_docling_pdf_indexing_with_pages(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = dataset["id"]
    pdf = (FIXTURES / "sample.pdf").read_bytes()
    with respx.mock(assert_all_called=False) as router:
        docling_routes(router)
        r = await upload(client, ds_id, "sample.pdf", pdf, index="true", options='{"do_ocr": false}')
        assert r.status_code == 201, r.text
        job = await wait_for_job(client, ds_id, r.json()["job"]["id"])
    assert job["status"] == "succeeded", job
    assert job["parser"] == "docling" and job["options"] == {"do_ocr": False}

    r = await client.post(f"/datasets/{ds_id}/search", json={"query": "terminate notice", "top_k": 1})
    top = r.json()["results"][0]
    assert top["page"] == 2 and top["section_path"] == ["Article 5. Termination"]

    # re-indexing with the same options uses the parse cache and does not call the convert API again
    with respx.mock(assert_all_called=False) as router:
        docling_routes(router)
        convert = router.routes[2]  # POST /v1/convert/file/async
        r = await client.post(
            f"/datasets/{ds_id}/documents/{top['document_id']}/index",
            json={"parser": "docling", "options": {"do_ocr": False}},
        )
        assert r.status_code == 202, r.text
        job = await wait_for_job(client, ds_id, r.json()["id"])
    assert job["status"] == "succeeded"
    assert convert.call_count == 0


async def test_docling_structure_chunking(client: httpx.AsyncClient, dataset: dict) -> None:
    """Chunks follow the DoclingDocument structure (recorded from the real parser on table.pdf),
    also when re-indexing from the parse cache."""
    ds_id = dataset["id"]
    md = (FIXTURES / "table_docling.md").read_text()
    structured = json.loads((FIXTURES / "table_docling.json").read_text())

    def routes(router: respx.MockRouter) -> None:
        docling_routes(router)
        result = json.loads((FIXTURES / "docling_result.json").read_text())
        result["document"].update(md_content=md, json_content=structured)
        router.get(f"{DOCLING_TEST_URL}/v1/result/t1").respond(json=result)

    with respx.mock(assert_all_called=False) as router:
        routes(router)
        r = await upload(client, ds_id, "table.pdf", (FIXTURES / "table.pdf").read_bytes(), index="true")
        assert r.status_code == 201, r.text
        doc_id = r.json()["document"]["id"]
        job = await wait_for_job(client, ds_id, r.json()["job"]["id"])
    assert job["status"] == "succeeded", job

    async def table_hit() -> dict:
        r = await client.post(f"/datasets/{ds_id}/search", json={"query": "egress price per region", "top_k": 4})
        return next(h for h in r.json()["results"] if "| ap-north |" in h["text"])

    hit = await table_hit()
    assert hit["page"] == 1 and hit["section_path"] == ["2. Price Table"]
    assert "Table 1. Monthly prices by region." in hit["text"] and "| us-east |" in hit["text"]
    assert (await client.get(f"/datasets/{ds_id}/documents/{doc_id}")).json()["num_chunks"] == 4

    # Re-index from the parse cache: the structure is read back from raw.json
    with respx.mock(assert_all_called=False) as router:
        routes(router)
        r = await client.post(f"/datasets/{ds_id}/documents/{doc_id}/index", json={"parser": "docling"})
        job = await wait_for_job(client, ds_id, r.json()["id"])
        assert router.routes[2].call_count == 0  # POST /v1/convert/file/async
    assert job["status"] == "succeeded", job
    assert (await client.get(f"/datasets/{ds_id}/documents/{doc_id}")).json()["num_chunks"] == 4
    assert (await table_hit())["section_path"] == ["2. Price Table"]


async def test_job_outliving_its_document_leaves_nothing(
    client: httpx.AsyncClient, dataset: dict, state: AppState
) -> None:
    """A job that keeps running after its document is deleted (e.g. an index request that arrived
    after the delete canceled the document's jobs) must not leave vectors or parse cache behind."""
    from rag_service.models.tables import Dataset, Document

    ds_id = dataset["id"]
    parse_done = False

    def poll(request: httpx.Request) -> httpx.Response:
        status = "success" if parse_done else "started"
        return httpx.Response(200, json={"task_id": "t1", "task_type": "convert", "task_status": status})

    with respx.mock(assert_all_called=False) as router:
        docling_routes(router)
        router.get(f"{DOCLING_TEST_URL}/v1/status/poll/t1").mock(side_effect=poll)
        r = await upload(client, ds_id, "late.pdf", (FIXTURES / "sample.pdf").read_bytes(), index="true")
        doc_id, job_id = r.json()["document"]["id"], r.json()["job"]["id"]
        for _ in range(200):
            if (await client.get(f"/datasets/{ds_id}/jobs/{job_id}")).json()["status"] == "running":
                break
            await asyncio.sleep(0.02)

        # The rest of a document delete, without canceling the job
        async with state.db.sessionmaker() as s:
            ds = await s.get(Dataset, ds_id)
            doc = await s.get(Document, doc_id)
            assert ds is not None and doc is not None
            storage_path = doc.storage_path
            await s.delete(doc)
            await s.commit()
        await state.vectors.delete_document(ds, doc_id)
        state.blob.delete_document(ds_id, doc_id, storage_path)

        parse_done = True
        await asyncio.wait_for(state.runner.wait_idle(), 30)

    assert await state.vectors.count_document_nodes(ds, doc_id) == 0
    assert not (state.blob.parsed / ds_id / doc_id).exists()


async def test_cancel_running_job(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = dataset["id"]
    with respx.mock(assert_all_called=False) as router:
        docling_routes(router, statuses=["started"])  # a conversion that never finishes
        r = await upload(client, ds_id, "slow.pdf", b"%PDF-1.4 slow", index="true")
        job_id = r.json()["job"]["id"]
        for _ in range(100):
            j = (await client.get(f"/datasets/{ds_id}/jobs/{job_id}")).json()
            if j["status"] == "running":
                break
            await asyncio.sleep(0.02)
        r = await client.post(f"/datasets/{ds_id}/jobs/{job_id}/cancel")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "canceled"
    doc = (await client.get(f"/datasets/{ds_id}/documents/{r.json()['document_id']}")).json()
    assert doc["status"] == "failed"  # document had no previous index
    r = await client.post(f"/datasets/{ds_id}/jobs/{job_id}/cancel")
    assert r.status_code == 409


async def test_bulk_index(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = dataset["id"]
    for i in range(3):
        await upload(client, ds_id, f"n{i}.txt", f"note number {i}".encode())
    await upload(client, ds_id, "bin.exe", b"MZ")
    r = await client.post(f"/datasets/{ds_id}/index", json={"parser": "docling"})
    assert r.status_code == 202, r.text
    out = r.json()
    assert len(out["jobs"]) == 3
    assert out["skipped"] == [
        {"document_id": out["skipped"][0]["document_id"], "reason": "unsupported_parser_or_format"}
    ]
    for j in out["jobs"]:
        assert (await wait_for_job(client, ds_id, j["id"]))["status"] == "succeeded"
    r = await client.get(f"/datasets/{ds_id}")
    assert r.json()["document_counts"]["indexed"] == 3


# ---------------- mineru (mocked service) ----------------


async def test_mineru_pdf_indexing(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = dataset["id"]
    pdf = (FIXTURES / "sample.pdf").read_bytes()
    with respx.mock(assert_all_called=False) as router:
        mock_mineru(router, base=MINERU_TEST_URL)
        r = await upload(client, ds_id, "sample.pdf", pdf, index="true", parser="mineru")
        assert r.status_code == 201, r.text
        assert r.json()["job"]["options"] == {"tier": "basic"}  # the server's default tier is recorded
        job = await wait_for_job(client, ds_id, r.json()["job"]["id"])
    assert job["status"] == "succeeded", job

    doc = (await client.get(f"/datasets/{ds_id}/documents/{job['document_id']}")).json()
    assert doc["parser"] == "mineru" and doc["parser_backend"] == "tier:basic"
    r = await client.post(f"/datasets/{ds_id}/search", json={"query": "terminate notice", "top_k": 1})
    top = r.json()["results"][0]
    assert top["page"] == 2 and top["section_path"] == ["Article 5. Termination"]


async def test_reindex_while_job_active_returns_409(client: httpx.AsyncClient, dataset: dict) -> None:
    ds_id = dataset["id"]
    with respx.mock(assert_all_called=False) as router:
        docling_routes(router, statuses=["started"])  # a conversion that never finishes
        r = await upload(client, ds_id, "busy.pdf", b"%PDF-1.4 busy", index="true")
        doc_id, job_id = r.json()["document"]["id"], r.json()["job"]["id"]
        r = await client.post(f"/datasets/{ds_id}/documents/{doc_id}/index", json={"parser": "docling"})
        assert r.status_code == 409, r.text
        assert r.json()["error"]["code"] == "job_in_progress"
        assert doc_id in r.json()["error"]["message"]
        await client.post(f"/datasets/{ds_id}/jobs/{job_id}/cancel")
