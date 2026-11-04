"""Embedding model construction.

A new instance is built per request from (catalog entry, API key). Keys are never put into
LlamaIndex's global `Settings`.

Implemented providers:
- `hash`: local embedding for development/tests. Works without external calls or API keys;
  hashes character n-grams so it reflects lexical similarity only (no semantic search quality).

The `litellm` provider (OpenAI-compatible, Bedrock models) will be added next.
"""

import hashlib
import math
import re
import unicodedata
from collections.abc import Callable
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import Field

from rag_service.providers.catalog import EmbeddingModelEntry

_WORD_RE = re.compile(r"\w+", re.UNICODE)


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


EmbeddingBuilder = Callable[[EmbeddingModelEntry, str | None], BaseEmbedding]


def _build_hash(entry: EmbeddingModelEntry, _api_key: str | None) -> BaseEmbedding:
    return HashEmbedding(model_name=entry.model, dim=entry.dim, embed_batch_size=256)


# provider name → (builder, whether an API key is required)
PROVIDERS: dict[str, tuple[EmbeddingBuilder, bool]] = {
    "hash": (_build_hash, False),
}


def provider_requires_api_key(provider: str) -> bool:
    return PROVIDERS[provider][1]


def build_embedding(entry: EmbeddingModelEntry, api_key: str | None, **_: Any) -> BaseEmbedding:
    builder, needs_key = PROVIDERS[entry.provider]
    if needs_key and not api_key:
        raise ValueError(f"embedding provider '{entry.provider}' requires an API key")
    return builder(entry, api_key)
