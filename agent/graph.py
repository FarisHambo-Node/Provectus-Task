"""Graph assembly and the public entry point for running a turn.

Shape:

    START
      └─> prepare_turn
            ├─(no question)──────────────> clarify ──> END
            └─> analyze_query
                  ├─(no tools / unsure)──> clarify ──> END
                  └─(fan out)────────────> rag_search ┐
                                           web_search ┴─> gather
                                                            ├─> synthesize ─> END
                                                            ├─> handle_failure ─> END
                                                            └─> clarify ─> END

Every branch is a conditional edge, so the routing policy is data (`state`) and
not control flow buried inside nodes.

Multi-turn: the compiled graph is invoked once per user message with a
`thread_id` in the config. The checkpointer reloads `messages`, `turn_index`,
and `summary` for that thread; `prepare_turn` clears the per-turn channels.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.container import AgentContainer, build_container
from agent.errors import AgentError
from agent.nodes.analyze import analyze_query
from agent.nodes.gather import gather
from agent.nodes.prepare_turn import prepare_turn
from agent.nodes.recovery import clarify, handle_failure
from agent.nodes.synthesize import synthesize
from agent.nodes.tool_nodes import rag_search_node, web_search_node
from agent.routing import (
    NODE_ANALYZE,
    NODE_CLARIFY,
    NODE_FAILURE,
    NODE_GATHER,
    NODE_PREPARE,
    NODE_SYNTHESIZE,
    route_after_analysis,
    route_after_gather,
    route_after_prepare,
)
from agent.state import AgentState, ToolName, TurnStatus, trace_for_turn

NODE_RAG = str(ToolName.RAG_SEARCH)
NODE_WEB = str(ToolName.WEB_SEARCH)


def build_graph(
    container: AgentContainer,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Wire the nodes and compile.

    TODO: swap `InMemorySaver` for a durable saver (DynamoDB, Postgres) before
    this runs behind more than one process — in-memory state dies with the
    Lambda container and breaks multi-turn for the next request.
    """
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node(NODE_PREPARE, partial(prepare_turn, container=container))
    builder.add_node(NODE_ANALYZE, partial(analyze_query, container=container))
    builder.add_node(NODE_RAG, partial(rag_search_node, container=container))
    builder.add_node(NODE_WEB, partial(web_search_node, container=container))
    builder.add_node(NODE_GATHER, partial(gather, container=container))
    builder.add_node(NODE_SYNTHESIZE, partial(synthesize, container=container))
    builder.add_node(NODE_CLARIFY, partial(clarify, container=container))
    builder.add_node(NODE_FAILURE, partial(handle_failure, container=container))

    builder.add_edge(START, NODE_PREPARE)
    builder.add_conditional_edges(
        NODE_PREPARE,
        route_after_prepare,
        [NODE_ANALYZE, NODE_CLARIFY],
    )
    builder.add_conditional_edges(
        NODE_ANALYZE,
        route_after_analysis,
        [NODE_RAG, NODE_WEB, NODE_CLARIFY],
    )
    # Tool branches converge on the join node; gather runs once, after both.
    builder.add_edge(NODE_RAG, NODE_GATHER)
    builder.add_edge(NODE_WEB, NODE_GATHER)
    builder.add_conditional_edges(
        NODE_GATHER,
        route_after_gather,
        [NODE_SYNTHESIZE, NODE_FAILURE, NODE_CLARIFY],
    )
    builder.add_edge(NODE_SYNTHESIZE, END)
    builder.add_edge(NODE_CLARIFY, END)
    builder.add_edge(NODE_FAILURE, END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


class ConversationAgent:
    """Façade over the compiled graph that owns per-thread invocation.

    Keeping this separate from `build_graph` means the HTTP handler, the CLI,
    and the tests all enter through one method with the same error handling.
    """

    def __init__(
        self,
        container: AgentContainer | None = None,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.container = container or build_container()
        self.graph = build_graph(self.container, checkpointer=checkpointer)

    async def ask(self, question: str, *, thread_id: str = "local") -> dict[str, Any]:
        """Run one turn. Returns a serialisable summary of the result.

        Raises only on programming errors; dependency failures come back as
        `status == failed` with an explanatory answer.
        """
        config = {"configurable": {"thread_id": thread_id}}
        inputs: AgentState = {
            "messages": [HumanMessage(content=question)],
            "thread_id": thread_id,
        }
        try:
            final: AgentState = await self.graph.ainvoke(inputs, config=config)
        except AgentError as exc:
            self.container.logger.error(
                "graph.turn_failed", thread_id=thread_id, error_code=exc.code
            )
            raise
        except Exception:
            self.container.logger.exception("graph.unhandled_error", thread_id=thread_id)
            raise

        return {
            "thread_id": thread_id,
            "turn_index": final.get("turn_index", 0),
            "status": str(final.get("status", TurnStatus.PENDING)),
            "answer": final.get("answer", ""),
            "citations": [c.model_dump() for c in final.get("citations", [])],
            "tools_used": [inv.tool for inv in final.get("invocations", []) if inv.ok],
            "failures": final.get("failures", []),
            "trace": [event.summary() for event in trace_for_turn(final)],
        }

    async def history(self, *, thread_id: str = "local") -> list[dict[str, str]]:
        """Replay the stored transcript for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.graph.aget_state(config)
        return [
            {"role": getattr(message, "type", "unknown"), "content": str(message.content)}
            for message in (snapshot.values or {}).get("messages", [])
        ]
