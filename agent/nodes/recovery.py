"""Terminal nodes for the paths that cannot produce a cited answer.

Both still append an `AIMessage`, so conversation state stays a well-formed
alternating transcript and the next turn can reference what happened.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agent.container import AgentContainer
from agent.nodes._common import node_span
from agent.observability import TraceEvent
from agent.state import AgentState, TurnStatus


def clarification_prompt(query: str) -> str:
    if not query:
        return "I did not get a question. What would you like me to look up?"
    return (
        f'I am not sure which sources would answer "{query}". '
        "Could you add the system, team, or time range you mean?"
    )


def failure_message(state: AgentState) -> str:
    """Explain the failure without leaking stack traces or internal hosts."""
    failures = state.get("failures", [])
    codes = sorted({str(failure.get("code", "unknown")) for failure in failures})
    if not failures:
        return (
            "I searched but found nothing relevant, so I would rather not guess. "
            "Try rephrasing, or ask me to widen the search."
        )
    detail = ", ".join(codes)
    return (
        "I could not reach the sources I need to answer this "
        f"({detail}), so I am not going to guess. Please retry in a moment."
    )


async def clarify(state: AgentState, container: AgentContainer) -> dict[str, Any]:
    trace: list[TraceEvent] = []
    async with node_span("clarify", state, container, trace) as attrs:
        message = clarification_prompt(state.get("query", ""))
        attrs["reason"] = "low_confidence_or_empty_query"
        container.metrics.increment("turn.clarification")
        return {
            "answer": message,
            "citations": [],
            "status": TurnStatus.NEEDS_CLARIFICATION,
            "messages": [AIMessage(content=message)],
            "trace": trace,
        }


async def handle_failure(state: AgentState, container: AgentContainer) -> dict[str, Any]:
    trace: list[TraceEvent] = []
    async with node_span("handle_failure", state, container, trace) as attrs:
        failures = state.get("failures", [])
        message = failure_message(state)
        attrs["failure_codes"] = [failure.get("code") for failure in failures]
        attrs["invocations"] = len(state.get("invocations", []))
        container.metrics.increment("turn.failed")
        container.logger.error(
            "turn.unanswerable",
            failure_codes=[failure.get("code") for failure in failures],
            retryable=[failure.get("retryable") for failure in failures],
        )
        return {
            "answer": message,
            "citations": [],
            "status": TurnStatus.FAILED,
            "messages": [AIMessage(content=message)],
            "trace": trace,
        }
