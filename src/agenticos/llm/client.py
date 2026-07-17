"""
Thin wrapper around the chat model backend.

Two things production code needs that the original prototype didn't have:

1. **Retries with backoff** on transient failures (Ollama warming up, a
   momentary connection blip) instead of failing on the first hiccup.
2. **A single choke point** for building the model, so swapping providers
   (Ollama -> a hosted API, a different local runtime) is a one-function
   change instead of a grep across the codebase.

`safe_invoke` / `safe_structured_invoke` return a `(result, error)` tuple
rather than raising, so callers (graph nodes) can degrade gracefully instead
of letting one bad LLM call crash the whole request — the same philosophy
the original prototype had, just with retries added before giving up.
"""
from __future__ import annotations

import time
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agenticos.logging_config import get_logger
from agenticos.settings import Settings, get_settings

logger = get_logger("llm.client")

T = TypeVar("T", bound=BaseModel)


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Builds the chat model used for planning and summarization.

    Backend is selected by `settings.llm_backend`:
      - "ollama" (default): local/self-hosted model, used for local dev/demo.
      - "groq": hosted OpenAI-compatible endpoint, used for public deployments
        where nothing is available to run Ollama continuously.

    Imports are local so environments that only need the graph's data-fetch
    path (e.g. some unit tests) aren't forced to have every LLM client
    library installed.
    """
    settings = settings or get_settings()

    if settings.llm_backend == "groq":
        from langchain_openai import ChatOpenAI

        if not settings.groq_api_key:
            raise ValueError("AGENTICOS_GROQ_API_KEY must be set when llm_backend=groq")
        return ChatOpenAI(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=settings.llm_temperature,
            timeout=settings.llm_request_timeout,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        # Previously configured in Settings but never wired through, so a
        # hung/unreachable Ollama server would block indefinitely instead of
        # failing fast into the retry path.
        client_kwargs={"timeout": settings.llm_request_timeout},
    )


def _retry_wait(attempt: int, settings: Settings) -> float:
    """Exponential backoff, capped at `llm_retry_max_wait`."""
    return min(settings.llm_retry_min_wait * (2 ** attempt), settings.llm_retry_max_wait)


def safe_invoke(
    model: BaseChatModel,
    messages: list[BaseMessage],
    settings: Settings | None = None,
) -> tuple[Any, str | None]:
    """Invokes a chat model, retrying transient errors with backoff.
    Returns (response, None) on success or (None, error_message) once
    retries are exhausted."""
    settings = settings or get_settings()
    last_error: Exception | None = None

    for attempt in range(settings.llm_max_retries + 1):
        try:
            return model.invoke(messages), None
        except Exception as exc:  # noqa: BLE001 - backend errors aren't a single stable type
            last_error = exc
            if attempt < settings.llm_max_retries:
                wait = _retry_wait(attempt, settings)
                logger.warning(
                    "LLM call failed, retrying",
                    extra={"extra_fields": {"attempt": attempt + 1, "wait_s": wait, "error": str(exc)}},
                )
                time.sleep(wait)

    logger.error("LLM call failed after retries", extra={"extra_fields": {"error": str(last_error)}})
    return None, str(last_error)


def safe_structured_invoke(
    model: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    settings: Settings | None = None,
) -> tuple[T | None, Any, str | None]:
    """Like `safe_invoke`, but binds the model to a structured-output schema
    (a Pydantic model) up front and returns a validated instance of it,
    rather than requiring every caller to hand-parse `tool_calls[0]["args"]`.

    Uses `include_raw=True` so the raw `AIMessage` (and its token-usage
    metadata) is still available to callers that want to track usage,
    alongside the parsed, validated object.

    Returns (parsed, raw_message, error).
    """
    structured_model = model.with_structured_output(schema, include_raw=True, method="function_calling")
    result, err = safe_invoke(structured_model, messages, settings)
    if err:
        return None, None, err

    parsing_error = result.get("parsing_error")
    if parsing_error:
        return None, result.get("raw"), f"Failed to parse structured output: {parsing_error}"

    return result.get("parsed"), result.get("raw"), None