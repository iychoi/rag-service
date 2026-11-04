"""Embedding model construction.

A dataset fixes its embedding model at creation (`EmbeddingSpec`): provider, model name, and options.
A new embedding instance is built per request with the caller's API key; keys are never put into
LlamaIndex's global `Settings`.

Providers:
- `litellm`: any embedding model registered in the LiteLLM proxy (OpenAI-compatible API) at the
  server-configured `LITELLM_BASE_URL`. Requires `X-Model-API-Key`.
- `hash`: local embedding for development/tests. No external calls or API key; hashes character
  n-grams, so it reflects lexical similarity only (no semantic search quality).
"""

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from pydantic import Field

Purpose = Literal["document", "query"]

PROVIDERS = ("litellm", "hash")
HASH_MODEL = "hash-ngram-v1"
HASH_DEFAULT_DIM = 512

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class EmbeddingSpec:
    """Embedding settings fixed on a dataset."""

    provider: str
    model: str
    options: dict[str, Any] = field(default_factory=dict)  # dimensions, input_type

    @property
    def dimensions(self) -> int | None:
        return self.options.get("dimensions")

    def input_type(self, purpose: Purpose) -> str | None:
        it = self.options.get("input_type") or {}
        return it.get(purpose)


def provider_requires_api_key(provider: str) -> bool:
    return provider == "litellm"


class HashEmbedding(BaseEmbedding):
    """Character n-gram feature-hashing embedding (signed, L2-normalized)."""

    dim: int = Field(gt=0)
    ngram_sizes: tuple[int, ...] = (2, 3)

    @classmethod
    def class_name(cls) -> str:
        return "HashEmbedding"

    def _features(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text).lower()
        feats: list[str] = []
        for word in _WORD_RE.findall(text):
            feats.append(f"w:{word}")
            padded = f" {word} "
            for n in self.ngram_sizes:
                feats.extend(f"c{n}:{padded[i : i + n]}" for i in range(len(padded) - n + 1))
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            # pgvector cosine distance is NaN for zero vectors, so fall back to a small constant vector
            return [1.0 / math.sqrt(self.dim)] * self.dim
        return [v / norm for v in vec]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)


def build_embedding(
    spec: EmbeddingSpec,
    *,
    purpose: Purpose,
    api_key: str | None,
    base_url: str | None,
    batch_size: int = 64,
    timeout_s: float = 60.0,
    max_retries: int = 2,
) -> BaseEmbedding:
    """Build an embedding model for documents (indexing) or queries (search).

    Raises ValueError if the provider is unknown or a required key/base URL is missing.
    """
    if spec.provider == "hash":
        return HashEmbedding(
            model_name=spec.model, dim=spec.dimensions or HASH_DEFAULT_DIM, embed_batch_size=256
        )
    if spec.provider == "litellm":
        if not base_url:
            raise ValueError("LITELLM_BASE_URL is not configured")
        if not api_key:
            raise ValueError("an API key is required for the litellm provider")
        extra: dict[str, Any] = {}
        if input_type := spec.input_type(purpose):
            # Asymmetric models (e.g. Cohere) need different input types for documents and queries
            extra["extra_body"] = {"input_type": input_type}
        return OpenAILikeEmbedding(
            model_name=spec.model,
            api_base=base_url,
            api_key=api_key,
            dimensions=spec.dimensions,
            additional_kwargs=extra,
            embed_batch_size=batch_size,
            timeout=timeout_s,
            max_retries=max_retries,
            reuse_client=False,  # per-request instances must not leak HTTP clients
        )
    raise ValueError(f"unknown embedding provider '{spec.provider}'")
