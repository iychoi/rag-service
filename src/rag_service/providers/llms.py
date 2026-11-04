"""LLM construction for LiteLLM's OpenAI-compatible API.

A new instance is built per request with the caller's key; keys are never put into LlamaIndex's
global `Settings`. The caller passes its own httpx client and closes it when done: LlamaIndex
does not close the OpenAI client after streaming, so this keeps per-request instances from
leaking connections in both the streaming and non-streaming paths.
"""

import httpx
from llama_index.llms.openai_like import OpenAILike


def build_llm(
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_output_tokens: int,
    timeout_s: float,
    context_window: int,
    http_client: httpx.AsyncClient,
) -> OpenAILike:
    return OpenAILike(
        model=model,
        api_base=base_url,
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=context_window,
        max_tokens=max_output_tokens,
        temperature=0.1,
        timeout=timeout_s,
        max_retries=1,
        async_http_client=http_client,
    )
