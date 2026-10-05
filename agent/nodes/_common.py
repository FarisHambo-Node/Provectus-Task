"""Shared plumbing for node functions.

`node_span` bundles the three things every node does identically: stamp the
correlation ids onto logs, time the node, and record a `TraceEvent` tagged with
the turn it belongs to.

Nodes put the same `trace` list they passed in onto their state update and then
return from inside the `async with`. The span appends its event on exit, which
happens before the node actually returns, so the update carries the event.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from agent.container import AgentContainer
from agent.observability import TraceEvent, log_context, span
from agent.state import AgentState


def turn_tags(state: AgentState) -> dict[str, Any]:
    return {
        "thread_id": state.get("thread_id", "local"),
        "turn_index": int(state.get("turn_index", 0)),
    }


@asynccontextmanager
async def node_span(
    name: str,
    state: AgentState,
    container: AgentContainer,
    trace: list[TraceEvent],
) -> AsyncIterator[dict[str, Any]]:
    tags = turn_tags(state)
    with log_context(node=name, **tags):
        async with span(
            f"node.{name}",
            kind="node",
            logger=container.logger,
            metrics=container.metrics,
            trace=trace,
            **tags,
        ) as attrs:
            yield attrs
