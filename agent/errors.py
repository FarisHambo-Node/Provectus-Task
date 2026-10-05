"""Error taxonomy for the agentic RAG system.

Every failure the graph can hit maps to one of these classes, so a node never
has to inspect a message string to decide whether a retry is worthwhile.
`retryable` is the single source of truth for the retry policy in
`agent.tools.resilience`.
"""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base class for every error raised inside the agent."""

    code: str = "agent_error"
    retryable: bool = False
    # Whether the graph can still produce a (degraded) answer for the user.
    recoverable: bool = True

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "recoverable": self.recoverable,
            "context": self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if not self.context:
            return self.message
        return f"{self.message} ({self.context})"


class ConfigurationError(AgentError):
    """Settings are missing or malformed. Fail fast at startup, never retry."""

    code = "configuration_error"
    recoverable = False


class BudgetExceededError(AgentError):
    """The turn consumed more tool calls / wall time than allowed."""

    code = "budget_exceeded"


class PlanningError(AgentError):
    """Query analysis could not produce a usable tool plan."""

    code = "planning_error"


class ThreadNotFoundError(AgentError):
    """The requested conversation thread or checkpoint does not exist."""

    code = "thread_not_found"


class CheckpointError(AgentError):
    """The conversation store could not be read or written."""

    code = "checkpoint_error"
    retryable = True


class SynthesisError(AgentError):
    """The answer generator failed or returned something unusable."""

    code = "synthesis_error"


class NoGroundingError(AgentError):
    """No tool produced evidence, so answering would mean hallucinating."""

    code = "no_grounding"


class ToolError(AgentError):
    """Base class for tool-level failures. Always carries the tool name."""

    code = "tool_error"

    def __init__(
        self,
        message: str,
        *,
        tool: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context={**(context or {}), "tool": tool})
        self.tool = tool


class ToolNotFoundError(ToolError):
    """The planner asked for a tool that is not in the registry."""

    code = "tool_not_found"


class ToolTimeoutError(ToolError):
    code = "tool_timeout"
    retryable = True


class ToolThrottledError(ToolError):
    """Bedrock / OpenSearch throttling. Back off and retry."""

    code = "tool_throttled"
    retryable = True


class ToolTransientError(ToolError):
    """Connection reset, 5xx, partial read: worth one more attempt."""

    code = "tool_transient"
    retryable = True


class ToolPermanentError(ToolError):
    """Bad request, auth failure, unknown index: retrying cannot help."""

    code = "tool_permanent"


class CircuitOpenError(ToolError):
    """The breaker for this tool is open, so the call was not even attempted."""

    code = "circuit_open"


class RetrievalError(ToolError):
    code = "retrieval_error"
    retryable = True


class WebSearchError(ToolError):
    code = "web_search_error"
    retryable = True


def classify_exception(exc: BaseException, *, tool: str) -> ToolError:
    """Wrap an arbitrary exception into the tool error taxonomy.

    The real implementation keys off `botocore.exceptions.ClientError` codes
    (`ThrottlingException`, `ModelTimeoutException`, `ValidationException`, ...);
    until Bedrock is wired in we only need the asyncio cases.
    """
    if isinstance(exc, ToolError):
        return exc
    if isinstance(exc, AgentError):
        return ToolPermanentError(exc.message, tool=tool, context=exc.context)
    if isinstance(exc, TimeoutError):
        return ToolTimeoutError("tool call timed out", tool=tool)
    if isinstance(exc, (ConnectionError, OSError)):
        return ToolTransientError(f"transport failure: {exc!r}", tool=tool)
    return ToolPermanentError(f"unhandled exception: {exc!r}", tool=tool)
