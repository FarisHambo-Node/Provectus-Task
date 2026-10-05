"""Conditional edge functions.

All routing lives here, as pure synchronous functions over state. They decide
nothing about *how* work happens, only *where* the graph goes next, which makes
the control flow testable with a plain dict as state.

`route_after_analysis` returns a list, which is how LangGraph fans out: every
name in the list runs in the same superstep.
"""

from __future__ import annotations

from agent.memory import should_summarize
from agent.state import AgentState, ToolName, TurnStatus, remaining_tool_budget

NODE_PREPARE = "prepare_turn"
NODE_SUMMARIZE = "summarize_history"
NODE_ANALYZE = "analyze_query"
NODE_GATHER = "gather"
NODE_SYNTHESIZE = "synthesize"
NODE_CLARIFY = "clarify"
NODE_FAILURE = "handle_failure"

TOOL_NODES: dict[ToolName, str] = {
    ToolName.RAG_SEARCH: "rag_search",
    ToolName.WEB_SEARCH: "web_search",
}

# Match the Settings defaults; the graph binds the configured values.
DEFAULT_SUMMARIZE_AFTER_TURNS = 8
DEFAULT_MAX_TOOL_CALLS = 4


def route_after_prepare(
    state: AgentState,
    *,
    summarize_after_turns: int = DEFAULT_SUMMARIZE_AFTER_TURNS,
) -> str:
    """Decide between clarifying, compacting memory, or planning.

    Summarisation happens before planning so the planner and the query
    rewriter see the compacted history rather than the raw transcript.
    """
    if not state.get("query", "").strip():
        return NODE_CLARIFY
    if should_summarize(
        state.get("messages", []), summarize_after_turns=summarize_after_turns
    ):
        return NODE_SUMMARIZE
    return NODE_ANALYZE


def route_after_analysis(
    state: AgentState,
    *,
    max_tool_calls_per_turn: int = DEFAULT_MAX_TOOL_CALLS,
) -> list[str]:
    """Fan out to the planned tools, or divert to clarification.

    Returning several node names runs them concurrently. They all converge on
    `gather`, which LangGraph executes once every branch has finished.

    This is also where the tool-call budget is enforced: the fan-out is clipped
    to whatever the turn can still afford, so the breaker trips before the
    calls are made rather than after.
    """
    if state.get("status") == TurnStatus.NEEDS_CLARIFICATION:
        return [NODE_CLARIFY]

    plan = state.get("plan")
    if plan is None or not plan.tools:
        return [NODE_CLARIFY]

    targets = [TOOL_NODES[tool] for tool in plan.tools if tool in TOOL_NODES]
    if not targets:
        return [NODE_CLARIFY]

    budget = remaining_tool_budget(state, max_tool_calls_per_turn)
    if budget <= 0:
        # Only reachable once a replan loop re-enters this edge with the
        # budget already spent. Refuse rather than answer ungrounded.
        return [NODE_FAILURE]
    return targets[:budget]


def route_after_gather(state: AgentState) -> str:
    """Synthesise only with evidence in hand; otherwise take the failure path."""
    status = state.get("status")
    if status == TurnStatus.NEEDS_CLARIFICATION:
        return NODE_CLARIFY
    if status == TurnStatus.FAILED:
        return NODE_FAILURE
    if not (state.get("chunks") or state.get("web_results")):
        # Belt and braces: never let a status mismatch reach the synthesiser,
        # which would have to invent an answer.
        return NODE_FAILURE
    return NODE_SYNTHESIZE
