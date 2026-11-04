"""Embedding access for API handlers and jobs: spec from a dataset, key checks, dimension probing."""

from typing import TYPE_CHECKING

import openai
from llama_index.core.base.embeddings.base import BaseEmbedding

from rag_service.errors import APIError
from rag_service.models.schemas import EmbeddingConfig
from rag_service.models.tables import Dataset
from rag_service.providers.embeddings import (
    HASH_DEFAULT_DIM,
    HASH_MODEL,
    EmbeddingSpec,
    Purpose,
    build_embedding,
    provider_requires_api_key,
)
from rag_service.providers.upstream import upstream_api_error

if TYPE_CHECKING:  # avoid an import cycle: state → jobs.runner → services.indexing → here
    from rag_service.state import AppState

# pgvector `vector` HNSW indexes support at most 2000 dimensions
MAX_EMBEDDING_DIM = 2000
_PROBE_TEXT = "dimension probe"


def spec_from_config(cfg: EmbeddingConfig) -> EmbeddingSpec:
    options: dict = {}
    if cfg.dimensions is not None:
        options["dimensions"] = cfg.dimensions
    if cfg.input_type is not None:
        options["input_type"] = cfg.input_type.model_dump()
    if cfg.provider == "hash":
        return EmbeddingSpec(provider="hash", model=HASH_MODEL, options=options)
    if not cfg.model:
        raise APIError(422, "embedding_model_required", "embedding.model is required for litellm")
    return EmbeddingSpec(provider=cfg.provider, model=cfg.model, options=options)


def spec_from_dataset(ds: Dataset) -> EmbeddingSpec:
    return EmbeddingSpec(
        provider=ds.embedding_provider, model=ds.embedding_model, options=ds.embedding_options
    )


def require_embedding_access(state: "AppState", provider: str, api_key: str | None) -> None:
    """Raise the API error for a missing LiteLLM base URL or key, if the provider needs them."""
    if not provider_requires_api_key(provider):
        return
    if not state.settings.litellm_base_url:
        raise APIError(503, "litellm_not_configured", "LITELLM_BASE_URL is not configured")
    if not api_key:
        raise APIError(400, "missing_model_api_key", "X-Model-API-Key header is required")


def embedder(state: "AppState", spec: EmbeddingSpec, purpose: Purpose, api_key: str | None) -> BaseEmbedding:
    require_embedding_access(state, spec.provider, api_key)
    s = state.settings
    return build_embedding(
        spec,
        purpose=purpose,
        api_key=api_key,
        base_url=s.litellm_base_url,
        batch_size=s.embed_batch_size,
        timeout_s=s.embedding_timeout_s,
        max_retries=s.litellm_max_retries,
    )


async def embed_query(state: "AppState", ds: Dataset, query: str, api_key: str | None) -> list[float]:
    model = embedder(state, spec_from_dataset(ds), "query", api_key)
    try:
        return await model.aget_query_embedding(query)
    except Exception as e:
        if (err := upstream_api_error(e, state.settings.embedding_timeout_s)) is not None:
            raise err from e
        raise


async def probe_dimension(state: "AppState", spec: EmbeddingSpec, api_key: str | None) -> int:
    """Embed a short text once to learn the model's real output dimension."""
    if spec.provider == "hash":
        dim = spec.dimensions or HASH_DEFAULT_DIM
    else:
        model = embedder(state, spec, "document", api_key)
        try:
            dim = len(await model.aget_text_embedding(_PROBE_TEXT))
        except Exception as e:
            err = upstream_api_error(e, state.settings.embedding_timeout_s)
            if err is None:
                raise
            if isinstance(e, openai.BadRequestError | openai.NotFoundError):
                # The gateway rejected the request itself: most likely an unknown model name
                raise APIError(422, "invalid_embedding_model", err.message) from e
            raise err from e
    if dim > MAX_EMBEDDING_DIM:
        raise APIError(
            422,
            "embedding_dim_too_large",
            f"model returns {dim}-dimensional vectors; at most {MAX_EMBEDDING_DIM} are supported "
            "(request a smaller `dimensions` if the model supports it)",
        )
    return dim
