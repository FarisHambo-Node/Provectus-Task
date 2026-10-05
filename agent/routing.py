"""Conditional edge functions.

All routing lives here, as pure synchronous functions over state. They decide
nothing about *how* work happens, only *where* the graph goes next, which makes
the control flow testable with a plain dict as state.

`route_after_analysis` returns a list, which is how LangGraph fans out: every
name in the list runs in the same superstep.
"""

from __future__ import annotations

from agent.state import AgentState, TurnStatus, ToolName

NODE_PREPARE = "prepare_turn"
NODE_ANALYZE = "analyze_query"
NODE_GATHER = "gather"
NODE_SYNTHESIZE = "synthesize"
NODE_CLARIFY = "clarify"
NODE_FAILURE = "handle_failure"

TOOL_NODES: dict[ToolName, str] = {
    ToolName.RAG_SEARCH: "rag_search",
    ToolName.WEB_SEARCH: "web_search",
}


def route_after_prepare(state: AgentState) -> str:
    """Skip planning when there is no question to plan for."""
    if not state.get("query", "").strip():
        return NODE_CLARIFY
    return NODE_ANALYZE


def route_after_analysis(state: AgentState) -> list[str]:
    """Fan out to the planned tools, or divert to clarification.

    Returning several node names runs them concurrently. They all converge on
    `gather`, which LangGraph executes once both branches have finished.
    """
    if state.get("status") == TurnStatus.NEEDS_CLARIFICATION:
        return [NODE_CLARIFY]

    plan = state.get("plan")
    if plan is None or not plan.tools:
        return [NODE_CLARIFY]

    targets = [TOOL_NODES[tool] for tool in plan.tools if tool in TOOL_NODES]
    return targets or [NODE_CLARIFY]


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
