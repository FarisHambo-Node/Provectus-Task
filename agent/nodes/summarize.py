"""Summarisation node: compress old turns and shrink the checkpoint.

Runs only when the stored transcript exceeds `summarize_after_turns`, chosen by
a conditional edge off `prepare_turn`. It writes a new `summary` and a batch of
`RemoveMessage` instructions, so the next checkpoint holds the window plus one
paragraph instead of the whole conversation.
"""

from __future__ import annotations

from typing import Any

from agent.container import AgentContainer
from agent.memory import count_turns, fold_summary, messages_to_drop, should_summarize
from agent.nodes._common import node_span
from agent.observability import TraceEvent
from agent.state import AgentState


async def summarize_history(state: AgentState, container: AgentContainer) -> dict[str, Any]:
    trace: list[TraceEvent] = []
    settings = container.settings
    async with node_span("summarize_history", state, container, trace) as attrs:
        messages = state.get("messages", [])
        previous = state.get("summary", "")
        older, removals = messages_to_drop(
            messages, window_turns=settings.history_window_turns
        )

        attrs["turns_before"] = count_turns(messages)
        attrs["folded_messages"] = len(older)
        attrs["removed_messages"] = len(removals)

        if not older:
            # Nothing old enough to fold; leave state untouched.
            attrs["summary_chars"] = len(previous)
            return {"trace": trace}

        summary = fold_summary(previous, older, max_chars=settings.summary_max_chars)
        attrs["summary_chars"] = len(summary)
        container.metrics.increment("memory.summarized", 1.0)
        container.logger.info(
            "memory.summarized",
            folded_messages=len(older),
            removed_messages=len(removals),
            summary_chars=len(summary),
        )

        update: dict[str, Any] = {"summary": summary, "trace": trace}
        if removals:
            update["messages"] = removals
        return update


def needs_summary(state: AgentState, *, summarize_after_turns: int) -> bool:
    """Routing predicate, kept next to the node that implements it."""
    return should_summarize(
        state.get("messages", []), summarize_after_turns=summarize_after_turns
    )
