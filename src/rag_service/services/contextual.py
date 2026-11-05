"""Contextual retrieval: an LLM writes a short context that situates each chunk in its document.

The context is embedded together with the chunk (and stored in the node's `context` metadata), so
a chunk such as "The fee is 0.09 USD." is also found by queries about the region or table it
belongs to. See https://www.anthropic.com/news/contextual-retrieval.

Calls go to LiteLLM's OpenAI-compatible API with the caller's key. The document is sent as the
first content block with `cache_control`, so models that support prompt caching (Claude) read it
from the cache for every chunk after the first. The first chunk of each part runs alone to write
the cache; the rest run concurrently.

Documents longer than CONTEXTUAL_DOCUMENT_TOKENS are given to the LLM in consecutive parts,
and each chunk is situated within its own part.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import openai

from rag_service.services.tokens import count_tokens

log = logging.getLogger(__name__)

CHUNK_PROMPT = """\
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the \
purposes of improving search retrieval of the chunk. Write it in the language of the document. \
Answer only with the succinct context and nothing else."""


@dataclass
class ContextUsage:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: openai.types.CompletionUsage | None) -> None:
        self.calls += 1
        if usage is None:
            return
        self.input_tokens += usage.prompt_tokens or 0
        self.output_tokens += usage.completion_tokens or 0
        details = usage.prompt_tokens_details
        if details is not None and details.cached_tokens:
            self.cached_input_tokens += details.cached_tokens


@dataclass
class ContextGenerator:
    base_url: str
    model: str
    api_key: str
    max_output_tokens: int
    document_tokens: int
    concurrency: int
    timeout_s: float
    max_retries: int
    usage: ContextUsage = field(default_factory=ContextUsage)

    async def generate(
        self,
        chunks: list[str],
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> list[str]:
        """One context per chunk, in order. Raises the `openai` SDK's errors."""
        contexts = [""] * len(chunks)
        done = 0
        parts = split_into_parts(chunks, self.document_tokens)
        sem = asyncio.Semaphore(max(1, self.concurrency))
        async with openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_s,
            max_retries=self.max_retries,
        ) as client:

            async def one(document: str, i: int) -> None:
                nonlocal done
                async with sem:
                    contexts[i] = await self._context(client, document, chunks[i])
                done += 1
                if on_progress is not None:
                    await on_progress(done)

            for n, indexes in enumerate(parts, start=1):
                document = "\n\n".join(chunks[i] for i in indexes)
                if len(parts) > 1:
                    document = f"(Part {n} of {len(parts)} of the document)\n\n{document}"
                document = f"<document>\n{document}\n</document>"
                await one(document, indexes[0])  # writes the prompt cache for the rest
                async with asyncio.TaskGroup() as tg:
                    for i in indexes[1:]:
                        tg.create_task(one(document, i))
        return contexts

    async def _context(self, client: openai.AsyncOpenAI, document: str, chunk: str) -> str:
        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        # LiteLLM passes cache_control through to Anthropic/Bedrock Claude models
                        {"type": "text", "text": document, "cache_control": {"type": "ephemeral"}},  # type: ignore[typeddict-unknown-key]
                        {"type": "text", "text": CHUNK_PROMPT.format(chunk=chunk)},
                    ],
                }
            ],
        )
        self.usage.add(response.usage)
        content = response.choices[0].message.content if response.choices else None
        return (content or "").strip()


def split_into_parts(chunks: list[str], max_tokens: int) -> list[list[int]]:
    """Group consecutive chunk indexes so each group's text stays within max_tokens (a group always
    has at least one chunk)."""
    parts: list[list[int]] = []
    used = 0
    for i, text in enumerate(chunks):
        tokens = count_tokens(text) + 1
        if parts and used + tokens <= max_tokens:
            parts[-1].append(i)
            used += tokens
        else:
            parts.append([i])
            used = tokens
    return parts


async def probe_chat_model(
    *, base_url: str, model: str, api_key: str, timeout_s: float, max_retries: int
) -> None:
    """Call the model once with a tiny prompt, so a wrong model name or key fails at dataset
    creation instead of in every indexing job. Raises the `openai` SDK's errors."""
    async with openai.AsyncOpenAI(
        base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=max_retries
    ) as client:
        await client.chat.completions.create(
            model=model, max_tokens=1, messages=[{"role": "user", "content": "Reply with OK."}]
        )
