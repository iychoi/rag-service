"""Search and query (retrieval + LLM answer) API."""

from typing import Annotated

from fastapi import APIRouter, Body

from rag_service.api.deps import DatasetDep, ModelAPIKey, Session, State
from rag_service.api.docs import errors
from rag_service.errors import APIError
from rag_service.models.schemas import QueryRequest, QueryResponse, SearchRequest, SearchResponse
from rag_service.services.answering import extract_citations, generate_answer, number_sources
from rag_service.services.retrieval import retrieve, validate_retrieval_request

router = APIRouter(tags=["search"])


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
    "LLM call needs it. Pick the answering model with `model` (default `claude-sonnet-5-5`; allowed "
    "by default: `claude-sonnet-5-5`, `claude-haiku-5-5`). If nothing is retrieved, the LLM is not "
    "called and `answer` is null.",
    responses=errors(
        {
            400: ["missing_model_api_key"],
            404: ["dataset_not_found"],
            422: ["validation_error", "top_k_too_large", "invalid_filter", "model_not_allowed"],
            502: ["upstream_auth_failed", "upstream_error", "upstream_unreachable"],
            503: ["litellm_not_configured"],
            504: ["upstream_timeout"],
        }
    ),
)
async def query(
    body: Annotated[
        QueryRequest,
        Body(
            openapi_examples={
                "default_model": {
                    "summary": "Use the server's default model",
                    "value": {
                        "query": "How can the agreement be terminated?",
                        "top_k": 5,
                        "filters": {"metadata": {"year": 2025}},
                    },
                },
                "choose_model": {
                    "summary": "Choose the model (faster, cheaper Haiku)",
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
) -> QueryResponse:
    settings = state.settings
    if not settings.litellm_base_url:
        raise APIError(503, "litellm_not_configured", "LITELLM_BASE_URL is not configured")
    model = body.model or settings.llm_model
    if model not in settings.llm_allowed_model_list:
        raise APIError(
            422,
            "model_not_allowed",
            f"model '{model}' is not allowed (allowed: {settings.llm_allowed_model_list})",
        )
    validate_retrieval_request(state, body.top_k, body.filters)
    if not api_key:
        raise APIError(400, "missing_model_api_key", "X-Model-API-Key header is required")

    results = await retrieve(state, session, ds, body.query, body.top_k, body.filters, api_key)
    sources = number_sources(results)
    if not sources:
        return QueryResponse(
            query=body.query, answer=None, model=None, citations=[], sources=[], usage=None
        )

    answer, usage = await generate_answer(
        query=body.query,
        sources=sources,
        base_url=settings.litellm_base_url,
        model=model,
        api_key=api_key,
        max_output_tokens=body.max_output_tokens or settings.llm_max_output_tokens,
        timeout_s=settings.llm_timeout_s,
        context_window=settings.llm_context_window,
    )
    return QueryResponse(
        query=body.query,
        answer=answer,
        model=model,
        citations=extract_citations(answer, len(sources)),
        sources=sources,
        usage=usage,
    )
