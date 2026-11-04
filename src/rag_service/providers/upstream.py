"""Map errors from LiteLLM (OpenAI-compatible API, via the `openai` SDK) to API errors.

Shared by embedding calls (dataset creation, indexing, search) and LLM calls (/query).
Messages never contain the caller's key; key-like strings echoed by the gateway are redacted.
"""

import re

import openai

from rag_service.errors import APIError

_KEY_LIKE = re.compile(r"sk-[A-Za-z0-9_\-]{4,}")


def redact(text: str) -> str:
    return _KEY_LIKE.sub("sk-***", text)


def upstream_api_error(e: Exception, timeout_s: float | None = None) -> APIError | None:
    """Translate an `openai` SDK exception into an APIError; None if it is not one."""
    if isinstance(e, openai.AuthenticationError | openai.PermissionDeniedError):
        return APIError(502, "upstream_auth_failed", "model provider rejected the API key")
    if isinstance(e, openai.APITimeoutError):
        limit = f" within {timeout_s:.0f}s" if timeout_s else ""
        return APIError(504, "upstream_timeout", f"model provider did not respond{limit}")
    if isinstance(e, openai.APIConnectionError):
        return APIError(502, "upstream_unreachable", "could not reach the model gateway")
    if isinstance(e, openai.APIStatusError):
        return APIError(
            502,
            "upstream_error",
            f"model gateway returned HTTP {e.status_code}: {_upstream_message(e)}",
        )
    if isinstance(e, openai.APIError):
        # e.g. an error event inside a streamed response
        return APIError(502, "upstream_error", f"model gateway error: {redact(str(e.message))[:300]}")
    return None


def _upstream_message(e: openai.APIStatusError) -> str:
    body = e.body if isinstance(e.body, dict) else {}
    err = body.get("error", body)
    msg = err.get("message") if isinstance(err, dict) else None
    return redact(str(msg or e.message))[:300]
