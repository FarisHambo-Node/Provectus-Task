"""Tool contract shared by every capability the agent can invoke.

`Tool.run` is a template method: subclasses implement `_execute` and get
timeouts, retries, the circuit breaker, metrics, and tracing for free. It is
also the error boundary of the system — `run` never raises. A broken tool
returns `ToolResult(ok=False, error=...)` so the graph can degrade instead of
dropping the turn.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agent.errors import AgentError, classify_exception
from agent.observability import MetricsSink, StructuredLogger, TraceEvent, get_logger, span
from agent.state import RetrievedChunk, ToolInvocation, ToolName, WebResult
from agent.tools.resilience import (
    CircuitBreaker,
    ResiliencePolicy,
    call_with_resilience,
)


class ToolSpec(BaseModel):
    """What the planner sees. Doubles as the Bedrock `toolConfig` source."""

    name: ToolName
    description: str
    # Lexical hints for the heuristic planner; the LLM planner uses `description`.
    keywords: tuple[str, ...] = ()
    # Run this tool when nothing else matched.
    is_default: bool = False
    cost_hint: str = "low"

    def to_bedrock_tool_spec(self) -> dict[str, Any]:
        """Shape Bedrock Converse expects under `toolConfig.tools[].toolSpec`."""
        return {
            "name": str(self.name),
            "description": self.description,
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Self-contained search query.",
                        }
                    },
                    "required": ["query"],
                }
            },
        }


class ToolRequest(BaseModel):
    """Everything a tool needs, with the correlation ids for logging."""

    query: str
    top_k: int = 5
    min_similarity: float = 0.0
    thread_id: str = "unknown"
    turn_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Uniform tool outcome. Carries evidence or a structured error, never both."""

    tool: ToolName
    ok: bool
    latency_ms: float = 0.0
    attempts: int = 1
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    web_results: list[WebResult] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    trace: list[TraceEvent] = Field(default_factory=list)

    @property
    def result_count(self) -> int:
        return len(self.chunks) + len(self.web_results)

    @property
    def is_empty(self) -> bool:
        return self.ok and self.result_count == 0

    def to_invocation(self) -> ToolInvocation:
        return ToolInvocation(
            tool=str(self.tool),
            ok=self.ok,
            attempts=self.attempts,
            latency_ms=self.latency_ms,
            result_count=self.result_count,
            error=self.error,
        )


class Tool(ABC):
    """Base class for all tools."""

    def __init__(
        self,
        *,
        spec: ToolSpec,
        policy: ResiliencePolicy | None = None,
        logger: StructuredLogger | None = None,
        metrics: MetricsSink | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.spec = spec
        self.policy = policy or ResiliencePolicy()
        self._log = logger or get_logger(f"agent.tools.{spec.name}")
        self._metrics = metrics
        self._rng = rng or random.Random()
        self._semaphore = asyncio.Semaphore(self.policy.max_concurrency)
        self._breaker = CircuitBreaker(
            name=str(spec.name),
            failure_threshold=self.policy.breaker_failure_threshold,
            reset_after_s=self.policy.breaker_reset_after_s,
        )

    @property
    def name(self) -> ToolName:
        return self.spec.name

    async def run(self, request: ToolRequest) -> ToolResult:
        """Execute the tool. Never raises; failures come back as `ok=False`."""
        trace: list[TraceEvent] = []
        try:
            async with span(
                f"tool.{self.name}",
                kind="tool",
                logger=self._log,
                metrics=self._metrics,
                trace=trace,
                tool=str(self.name),
                thread_id=request.thread_id,
                turn_index=request.turn_index,
            ) as attrs:
                result, attempts = await call_with_resilience(
                    lambda: self._execute(request),
                    tool=str(self.name),
                    policy=self.policy,
                    logger=self._log,
                    metrics=self._metrics,
                    breaker=self._breaker,
                    semaphore=self._semaphore,
                    rng=self._rng,
                )
                result.attempts = attempts
                result.latency_ms = 0.0  # filled in from the span below
                attrs["attempts"] = attempts
                attrs["result_count"] = result.result_count
                attrs["breaker"] = str(self._breaker.state)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            error = exc if isinstance(exc, AgentError) else classify_exception(exc, tool=str(self.name))
            self._log.error(
                "tool.failed",
                tool=str(self.name),
                error_code=error.code,
                recoverable=error.recoverable,
                exc_info=not isinstance(exc, AgentError),
            )
            return ToolResult(
                tool=self.name,
                ok=False,
                attempts=self.policy.max_attempts if error.retryable else 1,
                latency_ms=_span_duration(trace),
                error=error.to_dict(),
                trace=trace,
            )

        result.latency_ms = _span_duration(trace)
        result.trace = trace
        if result.is_empty:
            self._log.warning("tool.empty_result", tool=str(self.name), query_chars=len(request.query))
        return result

    @abstractmethod
    async def _execute(self, request: ToolRequest) -> ToolResult:
        """Do the actual work. Raise `AgentError` subclasses on failure."""

    async def health_check(self) -> bool:
        """Override when the tool has a backing dependency worth probing."""
        return self._breaker.state is not None


def _span_duration(trace: list[TraceEvent]) -> float:
    return round(trace[-1].duration_ms, 2) if trace else 0.0
