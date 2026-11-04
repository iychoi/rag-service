"""Integration test against a real LiteLLM proxy.

Run: LITELLM_BASE_URL=https://.../v1 LITELLM_API_KEY=sk-... [LITELLM_EMBED_MODEL=titan-embed-v2] \
     pytest -m integration tests/test_integration_llm.py -s
Skipped when either variable is missing. Costs a few hundred tokens per run.
"""

import os

import pytest

from rag_service.models.schemas import SearchResult
from rag_service.services.answering import extract_citations, generate_answer, number_sources

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("LITELLM_BASE_URL")
API_KEY = os.environ.get("LITELLM_API_KEY")
MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")


@pytest.mark.skipif(not (BASE_URL and API_KEY), reason="LITELLM_BASE_URL / LITELLM_API_KEY not set")
async def test_real_llm_answers_from_sources_with_citations() -> None:
    sources = number_sources(
        [
            SearchResult(
                score=0.7,
                text="이용자는 30일 전에 서면으로 통지하여 계약을 해지할 수 있다.",
                document_id="a" * 32,
                filename="contract.md",
                page=None,
                section_path=["이용 약관", "제5조 계약의 해지"],
                chunk_index=3,
                metadata={},
            ),
            SearchResult(
                score=0.4,
                text="회사의 고의 또는 과실로 손해가 발생한 경우 회사는 그 손해를 배상한다.",
                document_id="a" * 32,
                filename="contract.md",
                page=None,
                section_path=["이용 약관", "제7조 손해배상"],
                chunk_index=5,
                metadata={},
            ),
        ]
    )
    assert BASE_URL and API_KEY
    answer, usage = await generate_answer(
        query="계약은 어떻게 해지할 수 있나요?",
        sources=sources,
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
        max_output_tokens=300,
        timeout_s=120,
        context_window=200_000,
    )
    print(f"\n[{MODEL}] {answer}\nusage={usage}")
    assert "30" in answer
    assert 1 in extract_citations(answer, len(sources))
    assert usage.input_tokens and usage.output_tokens


EMBED_MODEL = os.environ.get("LITELLM_EMBED_MODEL")


@pytest.mark.skipif(
    not (BASE_URL and API_KEY and EMBED_MODEL),
    reason="LITELLM_BASE_URL / LITELLM_API_KEY / LITELLM_EMBED_MODEL not set",
)
async def test_real_embedding_model() -> None:
    from rag_service.providers.embeddings import EmbeddingSpec, build_embedding

    assert EMBED_MODEL
    spec = EmbeddingSpec(provider="litellm", model=EMBED_MODEL)
    doc = build_embedding(spec, purpose="document", api_key=API_KEY, base_url=BASE_URL)
    query = build_embedding(spec, purpose="query", api_key=API_KEY, base_url=BASE_URL)
    vecs = await doc.aget_text_embedding_batch(["계약 해지는 30일 전 서면 통지", "회의록: 예산 승인 대기"])
    q = await query.aget_query_embedding("계약을 해지하려면?")
    dim = len(q)
    print(f"\n[{EMBED_MODEL}] dim={dim}")
    assert dim <= 2000 and all(len(v) == dim for v in vecs)

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True)) / (
            sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5
        )

    assert cos(q, vecs[0]) > cos(q, vecs[1])
