"""Answer generation for /query: prompt with numbered sources, call the LLM, extract citations."""

import re

import openai
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from rag_service.errors import APIError
from rag_service.models.schemas import QuerySource, SearchResult, TokenUsage
from rag_service.providers.llms import build_llm

SYSTEM_PROMPT = """\
You answer questions using only the numbered sources provided by the user.
Rules:
- Base every statement on the sources. Do not use outside knowledge.
- Cite the sources that support each statement with bracketed numbers right after it, e.g. [1] or [2][3].
- If the sources do not contain the answer, say that the documents do not contain it. Do not guess.
- Answer in the same language as the question.
- Be concise."""

_CITATION = re.compile(r"\[(\d{1,3})\]")
_KEY_LIKE = re.compile(r"sk-[A-Za-z0-9_\-]{4,}")


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
    except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
        raise APIError(502, "upstream_auth_failed", "model provider rejected the API key") from e
    except openai.APITimeoutError as e:
        raise APIError(504, "llm_timeout", f"LLM did not respond within {timeout_s:.0f}s") from e
    except openai.APIConnectionError as e:
        raise APIError(502, "llm_unreachable", "could not reach the LLM gateway") from e
    except openai.APIStatusError as e:
        # Never echo the request (it may contain the key); the status and upstream message are safe.
        detail = _upstream_message(e)
        raise APIError(502, "llm_error", f"LLM gateway returned HTTP {e.status_code}: {detail}") from e

    answer = (response.message.content or "").strip()
    counts = response.additional_kwargs or {}
    usage = TokenUsage(
        input_tokens=counts.get("prompt_tokens"), output_tokens=counts.get("completion_tokens")
    )
    return answer, usage


def _upstream_message(e: openai.APIStatusError) -> str:
    body = e.body if isinstance(e.body, dict) else {}
    err = body.get("error", body)
    msg = err.get("message") if isinstance(err, dict) else None
    # Gateways sometimes echo (partial) keys in error messages
    return _KEY_LIKE.sub("sk-***", str(msg or e.message))[:300]
