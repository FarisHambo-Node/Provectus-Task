"""Tool invocation nodes.

Both nodes are the same five lines around a different tool, so they share
`_invoke_tool`. They run in parallel when the planner selects both; the state
reducers merge whatever each branch produced.

Nothing here raises. `Tool.run` already converts failures into a `ToolResult`
with `ok=False`, and the node turns that into a `failures` entry that
`route_after_gather` and the synthesiser can reason about.
"""

from __future__ import annotations

from typing import Any

from agent.container import AgentContainer
from agent.errors import ToolNotFoundError
from agent.nodes._common import node_span
from agent.observability import SKIPPED_ATTR, TraceEvent
from agent.state import AgentState, ToolName, ToolPlan
from agent.tools.base import ToolRequest


def build_request(state: AgentState, container: AgentContainer) -> ToolRequest:
    plan: ToolPlan | None = state.get("plan")
    return ToolRequest(
        query=(plan.search_query if plan and plan.search_query else state.get("query", "")),
        top_k=container.settings.top_k,
        min_similarity=container.settings.min_similarity,
        thread_id=state.get("thread_id", "local"),
        turn_index=int(state.get("turn_index", 0)),
    )


async def _invoke_tool(
    tool_name: ToolName,
    state: AgentState,
    container: AgentContainer,
) -> dict[str, Any]:
    trace: list[TraceEvent] = []
    async with node_span(str(tool_name), state, container, trace) as attrs:
        try:
            tool = container.registry.get(tool_name)
        except ToolNotFoundError as exc:
            attrs[SKIPPED_ATTR] = True
            container.logger.error("tool.missing", tool=str(tool_name))
            return {"failures": [exc.to_dict()], "trace": trace}

        result = await tool.run(build_request(state, container))
        trace.extend(result.trace)
        attrs["ok"] = result.ok
        attrs["attempts"] = result.attempts
        attrs["result_count"] = result.result_count

        update: dict[str, Any] = {
            "invocations": [result.to_invocation()],
            "trace": trace,
        }
        if result.ok:
            if result.chunks:
                update["chunks"] = result.chunks
            if result.web_results:
                update["web_results"] = result.web_results
        else:
            update["failures"] = [result.error or {"code": "unknown", "tool": str(tool_name)}]
        return update


async def rag_search_node(state: AgentState, container: AgentContainer) -> dict[str, Any]:
    return await _invoke_tool(ToolName.RAG_SEARCH, state, container)


async def web_search_node(state: AgentState, container: AgentContainer) -> dict[str, Any]:
    return await _invoke_tool(ToolName.WEB_SEARCH, state, container)
