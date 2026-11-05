"""Answer generation for /query: fit sources to a token budget, prompt with numbered sources, call
the LLM (whole or streamed), extract citations."""

import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import NoReturn

import httpx
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.utils import get_tokenizer

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


def count_tokens(text: str) -> int:
    """Token count with the tokenizer chunking uses (tiktoken, bundled with LlamaIndex).

    Claude's tokenizer differs, so this is an estimate of what the LLM bills.
    """
    return len(get_tokenizer()(text))


def select_sources(
    results: list[SearchResult],
    max_tokens: int,
    count: Callable[[str], int] = count_tokens,
) -> tuple[list[QuerySource], int]:
    """Number `results` (best first) and keep them while their prompt blocks fit in `max_tokens`.

    Stops at the first block that does not fit, so the sources stay the best-scoring prefix and
    their ids stay 1..n. The first source is always kept, even if it alone exceeds the budget.
    Returns (sources, number of results dropped).
    """
    sources = number_sources(results)
    used = 0
    for i, s in enumerate(sources):
        used += count(format_source(s))
        if i > 0 and used > max_tokens:
            return sources[:i], len(sources) - i
    return sources, 0


def format_source(s: QuerySource) -> str:
    where = [s.filename or "unknown file"]
    if s.page is not None:
        where.append(f"page {s.page}")
    if s.section_path:
        where.append(" > ".join(s.section_path))
    return f"[{s.id}] ({', '.join(where)})\n{s.text.strip()}"


def build_user_prompt(query: str, sources: list[QuerySource]) -> str:
    blocks = [format_source(s) for s in sources]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}"


def extract_citations(answer: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def build_messages(query: str, sources: list[QuerySource]) -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=build_user_prompt(query, sources)),
    ]


def _usage(counts: dict | None) -> TokenUsage | None:
    counts = counts or {}
    if counts.get("prompt_tokens") is None and counts.get("completion_tokens") is None:
        return None
    return TokenUsage(
        input_tokens=counts.get("prompt_tokens"), output_tokens=counts.get("completion_tokens")
    )


def _raise_upstream(e: Exception, timeout_s: float) -> NoReturn:
    if (err := upstream_api_error(e, timeout_s)) is not None:
        raise err from e
    raise e


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
    async with httpx.AsyncClient(timeout=timeout_s) as http:
        llm = build_llm(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
            context_window=context_window,
            http_client=http,
        )
        try:
            response = await llm.achat(build_messages(query, sources))
        except Exception as e:
            _raise_upstream(e, timeout_s)

    answer = (response.message.content or "").strip()
    return answer, _usage(response.additional_kwargs) or TokenUsage()


@dataclass
class StreamResult:
    """Filled in while the stream is consumed."""

    answer: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)


async def open_answer_stream(
    *,
    query: str,
    sources: list[QuerySource],
    base_url: str,
    model: str,
    api_key: str,
    max_output_tokens: int,
    timeout_s: float,
    context_window: int,
) -> tuple[AsyncIterator[str], StreamResult]:
    """Start a streaming answer and return (text deltas, result filled in as they are consumed).

    The first chunk is read before returning, so upstream errors that happen when the request is
    sent (bad key, unknown model, unreachable gateway) raise an APIError here, before any response
    bytes go out. Errors after that are raised from the iterator.
    """
    http = httpx.AsyncClient(timeout=timeout_s)
    llm = build_llm(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_output_tokens=max_output_tokens,
        timeout_s=timeout_s,
        context_window=context_window,
        http_client=http,
    )
    try:
        gen = await llm.astream_chat(
            build_messages(query, sources), stream_options={"include_usage": True}
        )
        first = await anext(gen, None)
    except Exception as e:
        await http.aclose()
        _raise_upstream(e, timeout_s)

    result = StreamResult()

    async def deltas() -> AsyncIterator[str]:
        item = first
        try:
            while item is not None:
                if (usage := _usage(item.additional_kwargs)) is not None:
                    result.usage = usage
                if item.delta:
                    result.answer += item.delta
                    yield item.delta
                item = await anext(gen, None)
        except Exception as e:
            _raise_upstream(e, timeout_s)
        finally:
            await gen.aclose()
            await http.aclose()

    return deltas(), result
