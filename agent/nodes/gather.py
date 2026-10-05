"""Join node: assess the evidence after the parallel tool branches finish.

Having an explicit join (rather than conditional edges straight off each tool
node) means the success/failure decision is made once, with all branches
visible. Without it, a turn where retrieval succeeded and web search failed
would route to two different nodes in the same superstep.

Deduping already happened in the state reducers, so this node only classifies.
"""

from __future__ import annotations

from typing import Any

from agent.container import AgentContainer
from agent.nodes._common import node_span
from agent.observability import TraceEvent
from agent.state import AgentState, TurnStatus


def classify_turn(state: AgentState) -> TurnStatus:
    """Decide whether the turn is answerable, degraded, or dead."""
    invocations = state.get("invocations", [])
    failures = state.get("failures", [])
    has_evidence = bool(state.get("chunks") or state.get("web_results"))

    if has_evidence:
        return TurnStatus.DEGRADED if failures else TurnStatus.OK
    if not invocations:
        # No tool ran at all: the planner found nothing to do.
        return TurnStatus.NEEDS_CLARIFICATION
    # Tools ran and produced nothing, so any answer would be ungrounded.
    return TurnStatus.FAILED


async def gather(state: AgentState, container: AgentContainer) -> dict[str, Any]:
    trace: list[TraceEvent] = []
    async with node_span("gather", state, container, trace) as attrs:
        status = classify_turn(state)
        chunks = state.get("chunks", [])
        web_results = state.get("web_results", [])
        failures = state.get("failures", [])

        attrs["chunks"] = len(chunks)
        attrs["web_results"] = len(web_results)
        attrs["failures"] = len(failures)
        attrs["status"] = str(status)
        attrs["top_score"] = chunks[0].score if chunks else None

        if status in (TurnStatus.DEGRADED, TurnStatus.FAILED):
            container.logger.warning(
                "turn.degraded",
                status=str(status),
                failure_codes=[failure.get("code") for failure in failures],
                chunks=len(chunks),
                web_results=len(web_results),
            )

        return {"status": status, "trace": trace}
