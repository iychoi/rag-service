"""Search and query (retrieval + LLM answer) API."""

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_service.api.deps import DatasetDep, ModelAPIKey, Session, State
from rag_service.api.docs import errors
from rag_service.errors import APIError
from rag_service.models.schemas import (
    QueryRequest,
    QueryResponse,
    QueryStreamDelta,
    QueryStreamDone,
    QueryStreamSources,
    QuerySource,
    SearchRequest,
    SearchResponse,
)
from rag_service.services.answering import (
    StreamResult,
    extract_citations,
    generate_answer,
    number_sources,
    open_answer_stream,
)
from rag_service.services.retrieval import retrieve, validate_retrieval_request

router = APIRouter(tags=["search"])

_SSE_EXAMPLE = (
    "event: sources\n"
    'data: {"query": "How can the agreement be terminated?", "sources": [{"id": 1, "score": 0.694, '
    '"text": "## Article 5. Termination\\n\\nEither party may terminate with 30 days written notice.", '
    '"document_id": "bbd713912b614848a8f0def3f73aa304", "filename": "contract.pdf", "page": 2, '
    '"section_path": ["Article 5. Termination"], "chunk_index": 1, "metadata": {"year": 2025}}]}\n\n'
    "event: delta\n"
    'data: {"text": "Either party may terminate"}\n\n'
    "event: delta\n"
    'data: {"text": " with 30 days written notice [1]."}\n\n'
    "event: done\n"
    'data: {"answer": "Either party may terminate with 30 days written notice [1].", '
    '"model": "claude-sonnet-5-5", "citations": [1], "usage": {"input_tokens": 412, "output_tokens": 15}}\n\n'
)


@router.post(
    "/datasets/{dataset_id}/search",
    response_model=SearchResponse,
    summary="Search a dataset",
    description="Returns the chunks most similar to the query from all indexed documents of the "
    "dataset. Send `X-Model-API-Key` if the dataset uses a `litellm` embedding model.",
    responses=errors(
        {
            400: ["missing_model_api_key"],
            404: ["dataset_not_found"],
            422: ["validation_error", "top_k_too_large", "invalid_filter"],
            502: ["upstream_auth_failed", "upstream_error", "upstream_unreachable"],
            503: ["litellm_not_configured"],
            504: ["upstream_timeout"],
        }
    ),
)
async def search(
    body: SearchRequest,
    ds: DatasetDep,
    state: State,
    session: Session,
    api_key: ModelAPIKey,
) -> SearchResponse:
    validate_retrieval_request(state, body.top_k, body.filters)
    results = await retrieve(state, session, ds, body.query, body.top_k, body.filters, api_key)
    return SearchResponse(query=body.query, results=results)


@router.post(
    "/datasets/{dataset_id}/query",
    response_model=QueryResponse,
    summary="Ask a question (retrieval + LLM answer)",
    description="Retrieves the most relevant chunks like /search, then asks the LLM (via LiteLLM) to "
    "answer only from them, citing sources as [n]. `X-Model-API-Key` is always required because the "
    "LLM call needs it. `model` (required) is any LiteLLM model name and is passed as-is; an unknown "
    "model returns LiteLLM's error as 502 upstream_error. If nothing "
    "is retrieved, the LLM is not called and `answer` is null.\n\n"
    "With `\"stream\": true` the response is `text/event-stream` (Server-Sent Events): `sources` "
    "(QueryStreamSources), then `delta` events (QueryStreamDelta) with answer text, then `done` "
    "(QueryStreamDone). Errors before streaming starts (validation, key, retrieval, the LLM rejecting "
    "the request) are normal JSON error responses; an error after streaming started is sent as an "
    "`error` event with the usual `{\"error\": {\"code\", \"message\"}}` body.",
    responses=errors(
        {
            400: ["missing_model_api_key"],
            404: ["dataset_not_found"],
            422: ["validation_error", "top_k_too_large", "invalid_filter"],
            502: ["upstream_auth_failed", "upstream_error", "upstream_unreachable"],
            503: ["litellm_not_configured"],
            504: ["upstream_timeout"],
        }
    )
    | {
        200: {
            "description": "The answer (JSON), or Server-Sent Events when `stream` is true",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": _SSE_EXAMPLE,
                }
            },
        }
    },
)
async def query(
    body: Annotated[
        QueryRequest,
        Body(
            openapi_examples={
                "sonnet": {
                    "summary": "Answer with Claude Sonnet",
                    "value": {
                        "query": "How can the agreement be terminated?",
                        "model": "claude-sonnet-5-5",
                        "top_k": 5,
                        "filters": {"metadata": {"year": 2025}},
                    },
                },
                "stream": {
                    "summary": "Stream the answer (Server-Sent Events)",
                    "value": {
                        "query": "How can the agreement be terminated?",
                        "model": "claude-sonnet-5-5",
                        "stream": True,
                    },
                },
                "haiku": {
                    "summary": "Faster, cheaper model with a shorter answer",
                    "value": {
                        "query": "계약 해지 조건은?",
                        "model": "claude-haiku-5-5",
                        "max_output_tokens": 300,
                    },
                },
            }
        ),
    ],
    ds: DatasetDep,
    state: State,
    session: Session,
    api_key: ModelAPIKey,
) -> QueryResponse | StreamingResponse:
    settings = state.settings
    if not settings.litellm_base_url:
        raise APIError(503, "litellm_not_configured", "LITELLM_BASE_URL is not configured")
    model = body.model  # passed to LiteLLM as-is; its errors are returned
    validate_retrieval_request(state, body.top_k, body.filters)
    if not api_key:
        raise APIError(400, "missing_model_api_key", "X-Model-API-Key header is required")

    results = await retrieve(state, session, ds, body.query, body.top_k, body.filters, api_key)
    sources = number_sources(results)
    llm_args: dict[str, Any] = dict(
        query=body.query,
        sources=sources,
        base_url=settings.litellm_base_url,
        model=model,
        api_key=api_key,
        max_output_tokens=body.max_output_tokens or settings.llm_max_output_tokens,
        timeout_s=settings.llm_timeout_s,
        context_window=settings.llm_context_window,
    )

    if body.stream:
        stream = None
        if sources:
            # Opens the upstream request now, so its errors become normal JSON responses
            stream = await open_answer_stream(**llm_args)
        return StreamingResponse(
            _sse_query(body.query, sources, model, stream),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if not sources:
        return QueryResponse(
            query=body.query, answer=None, model=None, citations=[], sources=[], usage=None
        )
    answer, usage = await generate_answer(**llm_args)
    return QueryResponse(
        query=body.query,
        answer=answer,
        model=model,
        citations=extract_citations(answer, len(sources)),
        sources=sources,
        usage=usage,
    )


def _sse(event: str, payload: BaseModel | dict[str, Any]) -> str:
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _sse_query(
    query_text: str,
    sources: list[QuerySource],
    model: str,
    stream: tuple[AsyncIterator[str], StreamResult] | None,
) -> AsyncIterator[str]:
    yield _sse("sources", QueryStreamSources(query=query_text, sources=sources))
    if stream is None:  # nothing retrieved: the LLM is not called
        yield _sse("done", QueryStreamDone(answer=None, model=None, citations=[], usage=None))
        return
    deltas, result = stream
    try:
        async for text in deltas:
            yield _sse("delta", QueryStreamDelta(text=text))
    except APIError as e:
        yield _sse("error", {"error": {"code": e.code, "message": e.message}})
        return
    except Exception:  # noqa: BLE001 - the HTTP status is already sent; report in-band
        yield _sse("error", {"error": {"code": "internal_error", "message": "answer stream failed"}})
        return
    answer = result.answer.strip()
    yield _sse(
        "done",
        QueryStreamDone(
            answer=answer,
            model=model,
            citations=extract_citations(answer, len(sources)),
            usage=result.usage,
        ),
    )
