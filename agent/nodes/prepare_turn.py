"""Entry node: establish the turn and clear the previous turn's scratch state.

This is the seam between conversation state (persisted by the checkpointer) and
turn state (rebuilt for every question). Doing it in its own node lets the rest
of the graph assume `query` is set and the evidence channels are empty.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage

from agent.container import AgentContainer
from agent.nodes._common import node_span
from agent.observability import TraceEvent
from agent.state import AgentState, TurnStatus, new_turn_scratch


def latest_user_query(messages: list[AnyMessage]) -> str:
    """Last human turn, or empty string when there is nothing to answer."""
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            return str(message.content).strip()
    return ""


def recent_history(messages: list[AnyMessage], *, window_turns: int) -> list[AnyMessage]:
    """Keep the last `window_turns` exchanges verbatim.

    Older turns belong in `summary`; trimming here is what stops a long thread
    from crowding retrieved chunks out of the context window.
    """
    return list(messages or [])[-(window_turns * 2) :]


async def prepare_turn(state: AgentState, container: AgentContainer) -> dict[str, Any]:
    thread_id = state.get("thread_id") or "local"
    turn_index = int(state.get("turn_index", 0)) + 1
    trace: list[TraceEvent] = []
    # node_span tags events with the turn, so hand it the incremented value.
    scoped: AgentState = {**state, "thread_id": thread_id, "turn_index": turn_index}

    async with node_span("prepare_turn", scoped, container, trace) as attrs:
        messages = state.get("messages", [])
        query = latest_user_query(messages)
        history = recent_history(
            messages, window_turns=container.settings.history_window_turns
        )
        attrs["query_chars"] = len(query)
        attrs["history_messages"] = len(history)
        attrs["empty_query"] = not query

        update: dict[str, Any] = {
            **new_turn_scratch(),
            "thread_id": thread_id,
            "turn_index": turn_index,
            "query": query,
            "trace": trace,
        }
        if not query:
            update["status"] = TurnStatus.NEEDS_CLARIFICATION
        return update
