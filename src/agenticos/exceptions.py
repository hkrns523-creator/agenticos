"""Domain-specific exceptions.

Keeping these separate from generic `Exception` catches lets callers (and
tests) distinguish "the LLM backend is unreachable" from "the database file
is missing" from "the planner returned something we can't use" — each of
which the graph should handle differently instead of a single blanket
except-and-continue.
"""


class AgenticOSError(Exception):
    """Base class for all AgenticOS-specific errors."""


class LLMUnavailableError(AgenticOSError):
    """Raised when the LLM backend cannot be reached after retries."""


class ToolExecutionError(AgenticOSError):
    """Raised when a specialist tool (DB query, RAG lookup, etc.) fails."""


class PlannerParseError(AgenticOSError):
    """Raised when the planner's structured output can't be parsed/validated."""


class ConversationMemoryError(AgenticOSError):
    """Raised when reading or writing conversation history fails. Memory is
    treated as best-effort: callers should log and continue rather than let
    this abort an otherwise-successful investigation."""
