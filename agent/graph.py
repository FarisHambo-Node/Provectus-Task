"""Graph assembly and the public entry point for running a conversation.

Shape:

    START
      └─> prepare_turn
            ├─(no question)──────────────> clarify ──> END
            ├─(history too long)────────── summarize_history ─┐
            └─────────────────────────────────────────────────┴─> analyze_query
                  ├─(no tools / unsure)──> clarify ──> END
                  └─(fan out)────────────> rag_search ┐
                                           web_search ┴─> gather
                                                            ├─> synthesize ─> END
                                                            ├─> handle_failure ─> END
                                                            └─> clarify ─> END

Every branch is a conditional edge, so routing policy is data (`state`) rather
than control flow buried inside nodes.

Multi-turn works through the checkpointer. Each user message is one `ainvoke`
with a `thread_id` in the config: LangGraph reloads that thread's `messages`,
`turn_index`, and `summary`, `prepare_turn` clears the per-turn channels, and
the new values are written back at the end of the turn. Nothing is passed
between turns by the caller.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.container import AgentContainer, build_container
from agent.errors import AgentError, ThreadNotFoundError
from agent.memory import count_turns
from agent.nodes.analyze import analyze_query
from agent.nodes.gather import gather
from agent.nodes.prepare_turn import prepare_turn
from agent.nodes.recovery import clarify, handle_failure
from agent.nodes.summarize import summarize_history
from agent.nodes.synthesize import synthesize
from agent.nodes.tool_nodes import rag_search_node, web_search_node
from agent.persistence import checkpointer_scope
from agent.routing import (
    NODE_ANALYZE,
    NODE_CLARIFY,
    NODE_FAILURE,
    NODE_GATHER,
    NODE_PREPARE,
    NODE_SUMMARIZE,
    NODE_SYNTHESIZE,
    route_after_analysis,
    route_after_gather,
    route_after_prepare,
)
from agent.settings import Settings
from agent.state import AgentState, ToolName, TurnStatus, trace_for_turn

NODE_RAG = str(ToolName.RAG_SEARCH)
NODE_WEB = str(ToolName.WEB_SEARCH)


def build_graph(
    container: AgentContainer,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Wire the nodes and compile against a checkpointer.

    Defaults to `InMemorySaver`, which is correct for tests and a single-shot
    CLI and wrong for anything else; `ConversationAgent.session` builds the
    configured durable store instead.
    """
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node(NODE_PREPARE, partial(prepare_turn, container=container))
    builder.add_node(NODE_SUMMARIZE, partial(summarize_history, container=container))
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
        partial(
            route_after_prepare,
            summarize_after_turns=container.settings.summarize_after_turns,
        ),
        [NODE_SUMMARIZE, NODE_ANALYZE, NODE_CLARIFY],
    )
    builder.add_edge(NODE_SUMMARIZE, NODE_ANALYZE)
    builder.add_conditional_edges(
        NODE_ANALYZE,
        partial(
            route_after_analysis,
            max_tool_calls_per_turn=container.settings.max_tool_calls_per_turn,
        ),
        [NODE_RAG, NODE_WEB, NODE_CLARIFY, NODE_FAILURE],
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
    and the tests all enter through the same methods with the same error
    handling, and thread bookkeeping lives in one place.
    """

    def __init__(
        self,
        container: AgentContainer | None = None,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.container = container or build_container()
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(self.container, checkpointer=self.checkpointer)

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def session(
        cls,
        container: AgentContainer | None = None,
        *,
        settings: Settings | None = None,
    ) -> AsyncIterator["ConversationAgent"]:
        """Open the configured checkpoint store and build an agent on it.

        Use this in anything long-lived: the SQLite backend holds a connection
        that has to be closed, and the store should be opened once per process
        rather than once per request.
        """
        container = container or build_container(settings)
        async with checkpointer_scope(
            container.settings,
            logger=container.logger,
            metrics=container.metrics,
        ) as checkpointer:
            yield cls(container, checkpointer=checkpointer)

    # -- running a turn ----------------------------------------------------

    async def ask(
        self,
        question: str,
        *,
        thread_id: str = "local",
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """Run one turn. Returns a serialisable summary of the result.

        Raises only on programming errors; dependency failures come back as
        `status == failed` with an explanatory answer. Passing `checkpoint_id`
        resumes from that point in the thread instead of its current head.
        """
        config = self._config(thread_id, checkpoint_id)
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

        return self._summarize_turn(thread_id, final)

    # -- conversation state ------------------------------------------------

    async def history(self, *, thread_id: str = "local") -> list[dict[str, str]]:
        """The stored transcript for a thread, as reloaded from the checkpoint."""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        return [
            {"role": getattr(message, "type", "unknown"), "content": str(message.content)}
            for message in (snapshot.values or {}).get("messages", [])
        ]

    async def state(self, *, thread_id: str = "local") -> dict[str, Any]:
        """What the agent currently remembers about a thread."""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        values = snapshot.values or {}
        return {
            "thread_id": thread_id,
            "exists": bool(values),
            "turn_index": values.get("turn_index", 0),
            "stored_turns": count_turns(values.get("messages", [])),
            "stored_messages": len(values.get("messages", [])),
            "summary": values.get("summary", ""),
            "checkpoint_id": (snapshot.config or {})
            .get("configurable", {})
            .get("checkpoint_id"),
            "next_nodes": list(snapshot.next or ()),
        }

    async def checkpoints(
        self, *, thread_id: str = "local", limit: int = 20
    ) -> list[dict[str, Any]]:
        """List a thread's checkpoints, newest first.

        This is the handle for time travel: feed a `checkpoint_id` back into
        `ask` or `fork` to replay the conversation from that point.
        """
        entries: list[dict[str, Any]] = []
        async for snapshot in self.graph.aget_state_history(
            self._config(thread_id), limit=limit
        ):
            values = snapshot.values or {}
            entries.append(
                {
                    "checkpoint_id": (snapshot.config or {})
                    .get("configurable", {})
                    .get("checkpoint_id"),
                    "turn_index": values.get("turn_index", 0),
                    "status": str(values.get("status", "")),
                    "next_nodes": list(snapshot.next or ()),
                    "stored_messages": len(values.get("messages", [])),
                    "created_at": snapshot.created_at,
                }
            )
        return entries

    async def fork(
        self,
        question: str,
        *,
        thread_id: str,
        checkpoint_id: str,
        into_thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Branch a conversation: ask a different question from a past point.

        Without `into_thread_id` the branch is appended to the same thread,
        which is what "edit my last question" does in a chat UI.
        """
        if into_thread_id is None:
            return await self.ask(
                question, thread_id=thread_id, checkpoint_id=checkpoint_id
            )
        await self._copy_thread(thread_id, into_thread_id, checkpoint_id)
        return await self.ask(question, thread_id=into_thread_id)

    async def delete_thread(self, *, thread_id: str) -> None:
        """Forget a conversation. The right-to-erasure hook."""
        await self.checkpointer.adelete_thread(thread_id)
        self.container.logger.info("conversation.deleted", thread_id=thread_id)

    # -- internals ---------------------------------------------------------

    def _config(self, thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        configurable: dict[str, Any] = {"thread_id": thread_id}
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable}

    def _summarize_turn(self, thread_id: str, final: AgentState) -> dict[str, Any]:
        return {
            "thread_id": thread_id,
            "turn_index": final.get("turn_index", 0),
            "status": str(final.get("status", TurnStatus.PENDING)),
            "answer": final.get("answer", ""),
            "citations": [c.model_dump() for c in final.get("citations", [])],
            "tools_used": [inv.tool for inv in final.get("invocations", []) if inv.ok],
            "tool_calls_used": final.get("tool_calls_used", 0),
            "failures": final.get("failures", []),
            "summary": final.get("summary", ""),
            "stored_turns": count_turns(final.get("messages", [])),
            "trace": [event.summary() for event in trace_for_turn(final)],
        }

    async def _copy_thread(self, source: str, target: str, checkpoint_id: str) -> None:
        """Seed `target` with `source`'s state as of `checkpoint_id`.

        Replays the stored channel values into a fresh thread. Done through the
        graph rather than the saver's own copy helper so it works on every
        backend.
        """
        snapshot = await self.graph.aget_state(self._config(source, checkpoint_id))
        if not snapshot.values:
            raise ThreadNotFoundError(
                "no checkpoint to fork from",
                context={"thread_id": source, "checkpoint_id": checkpoint_id},
            )
        await self.graph.aupdate_state(
            self._config(target),
            {
                "messages": snapshot.values.get("messages", []),
                "summary": snapshot.values.get("summary", ""),
                "turn_index": snapshot.values.get("turn_index", 0),
            },
        )
        self.container.logger.info(
            "conversation.forked",
            source_thread=source,
            target_thread=target,
            checkpoint_id=checkpoint_id,
        )
