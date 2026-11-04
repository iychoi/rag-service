"""Answer generation for /query: prompt with numbered sources, call the LLM, extract citations."""

import re

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from rag_service.models.schemas import QuerySource, SearchResult, TokenUsage
from rag_service.providers.llms import build_llm
from rag_service.providers.upstream import upstream_api_error

SYSTEM_PROMPT = """\
You answer questions using only the numbered sources provided by the user.
Rules:
- Base every statement on the sources. Do not use outside knowledge.
- Cite the sources that support each statement with bracketed numbers right after it, e.g. [1] or [2][3].
- If the sources do not contain the answer, say that the documents do not contain it. Do not guess.
- Answer in the same language as the question.
- Be concise."""

_CITATION = re.compile(r"\[(\d{1,3})\]")


def number_sources(results: list[SearchResult]) -> list[QuerySource]:
    return [QuerySource(id=i, **r.model_dump()) for i, r in enumerate(results, start=1)]


def build_user_prompt(query: str, sources: list[QuerySource]) -> str:
    blocks = []
    for s in sources:
        where = [s.filename or "unknown file"]
        if s.page is not None:
            where.append(f"page {s.page}")
        if s.section_path:
            where.append(" > ".join(s.section_path))
        blocks.append(f"[{s.id}] ({', '.join(where)})\n{s.text.strip()}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}"


def extract_citations(answer: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


async def generate_answer(
    *,
    query: str,
    sources: list[QuerySource],
    base_url: str,
    model: str,
    api_key: str,
    max_output_tokens: int,
    timeout_s: float,
    context_window: int,
) -> tuple[str, TokenUsage]:
    llm = build_llm(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_output_tokens=max_output_tokens,
        timeout_s=timeout_s,
        context_window=context_window,
    )
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=build_user_prompt(query, sources)),
    ]
    try:
        response = await llm.achat(messages)
    except Exception as e:
        if (err := upstream_api_error(e, timeout_s)) is not None:
            raise err from e
        raise

    answer = (response.message.content or "").strip()
    counts = response.additional_kwargs or {}
    usage = TokenUsage(
        input_tokens=counts.get("prompt_tokens"), output_tokens=counts.get("completion_tokens")
    )
    return answer, usage

