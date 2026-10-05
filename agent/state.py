"""Graph state and the data types that flow through it.

Two categories of state, and the difference matters for multi-turn:

* **Conversation state** survives turns: `messages`, `summary`, `turn_index`.
  The checkpointer persists it per `thread_id`.
* **Turn scratch state** is per-question: `plan`, `chunks`, `web_results`,
  `citations`, `answer`, `trace`. `prepare_turn` clears it by writing `None`,
  which the `accumulate` reducer interprets as a reset.

The accumulating lists use reducers because tool nodes run in parallel: each
branch returns only its own slice and LangGraph merges them.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, TypedDict, TypeVar

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agent.observability import TraceEvent

T = TypeVar("T")


class ToolName(StrEnum):
    """Tool identifiers. Node names in the graph match these values."""

    RAG_SEARCH = "rag_search"
    WEB_SEARCH = "web_search"


class TurnStatus(StrEnum):
    PENDING = "pending"
    OK = "ok"
    # Answered, but at least one planned tool failed or returned nothing.
    DEGRADED = "degraded"
    NEEDS_CLARIFICATION = "needs_clarification"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


class RetrievedChunk(BaseModel):
    """One chunk from the internal document store."""

    chunk_id: str
    doc_id: str
    title: str
    source_uri: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    section: str | None = None


class WebResult(BaseModel):
    """One hit from the external web search tool."""

    url: str
    title: str
    snippet: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    published_at: str | None = None
    provider: str = "simulated"


class Citation(BaseModel):
    """A numbered reference the synthesiser is allowed to cite."""

    marker: str
    origin: str
    title: str
    locator: str
    snippet: str
    score: float | None = None


class ToolInvocation(BaseModel):
    """Audit record of one tool call. Kept even when the call failed."""

    tool: str
    ok: bool
    attempts: int = 1
    latency_ms: float = 0.0
    result_count: int = 0
    error: dict[str, Any] | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolPlan(BaseModel):
    """Output of query analysis: which tools to run and why."""

    tools: list[ToolName] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    # Follow-ups such as "what about the second one?" get resolved against
    # history here, so tools always receive a self-contained query.
    search_query: str = ""


# --------------------------------------------------------------------------- #
# Reducers
# --------------------------------------------------------------------------- #


def accumulate(left: list[T] | None, right: list[T] | None) -> list[T]:
    """Append-only list reducer where a `None` write clears the channel.

    Parallel tool nodes each append their own results; `prepare_turn` writes
    `None` to drop the previous turn's evidence.
    """
    if right is None:
        return []
    return [*(left or []), *right]


def dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the highest-scoring copy of each chunk, best first.

    Duplicates are normal — overlapping chunks, and a retry whose first attempt
    partially succeeded — and a duplicate in the prompt wastes tokens and skews
    the citation list.
    """
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        current = best.get(chunk.chunk_id)
        if current is None or chunk.score > current.score:
            best[chunk.chunk_id] = chunk
    return sorted(best.values(), key=lambda chunk: chunk.score, reverse=True)


def dedupe_web_results(results: list[WebResult]) -> list[WebResult]:
    best: dict[str, WebResult] = {}
    for result in results:
        current = best.get(result.url)
        if current is None or result.score > current.score:
            best[result.url] = result
    return sorted(best.values(), key=lambda result: result.score, reverse=True)


def merge_chunks(
    left: list[RetrievedChunk] | None, right: list[RetrievedChunk] | None
) -> list[RetrievedChunk]:
    """Reducer for retrieved chunks: merge parallel writes, dedupe, rank."""
    if right is None:
        return []
    return dedupe_chunks([*(left or []), *right])


def merge_web_results(
    left: list[WebResult] | None, right: list[WebResult] | None
) -> list[WebResult]:
    if right is None:
        return []
    return dedupe_web_results([*(left or []), *right])


def replace(left: T | None, right: T | None) -> T | None:
    """Last-write-wins. Explicit so parallel writes are an obvious choice."""
    return right


def add_counts(left: int | None, right: int | None) -> int:
    """Summing reducer for per-turn counters; a `None` write resets to zero.

    Summing rather than replacing is what makes the tool-call budget correct
    when branches run in parallel: each tool node reports its own single call
    and the channel ends up with the real total.
    """
    if right is None:
        return 0
    return (left or 0) + right


# The trace spans the whole thread, not one turn, because the interesting
# questions in a multi-turn session ("why did turn 3 stop using retrieval?")
# need the earlier turns. Bounded so a long thread cannot grow without limit.
MAX_TRACE_EVENTS = 200


def accumulate_trace(
    left: list[TraceEvent] | None, right: list[TraceEvent] | None
) -> list[TraceEvent]:
    if right is None:
        return []
    merged = [*(left or []), *right]
    return merged[-MAX_TRACE_EVENTS:]


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


class AgentState(TypedDict, total=False):
    """The single channel set for the whole graph."""

    # -- conversation (persisted across turns by the checkpointer) ----------
    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    turn_index: int
    # Rolling summary of turns older than `history_window_turns`.
    summary: str

    # -- turn scratch ------------------------------------------------------
    query: str
    plan: Annotated[ToolPlan | None, replace]
    chunks: Annotated[list[RetrievedChunk], merge_chunks]
    web_results: Annotated[list[WebResult], merge_web_results]
    invocations: Annotated[list[ToolInvocation], accumulate]
    failures: Annotated[list[dict[str, Any]], accumulate]
    citations: Annotated[list[Citation], replace]
    answer: str
    status: TurnStatus
    # Spend against the per-turn tool budget. See `remaining_tool_budget`.
    tool_calls_used: Annotated[int, add_counts]
    # Replan bookkeeping: which tools this turn already tried, and how many
    # times it has looped back to the planner.
    attempted_tools: Annotated[list[ToolName], accumulate]
    replan_count: Annotated[int, add_counts]

    # -- thread-level diagnostics -----------------------------------------
    trace: Annotated[list[TraceEvent], accumulate_trace]


def new_turn_scratch() -> dict[str, Any]:
    """Channel writes that reset per-turn state. Used by `prepare_turn`."""
    return {
        "plan": None,
        "chunks": None,
        "web_results": None,
        "invocations": None,
        "failures": None,
        "citations": [],
        "answer": "",
        "status": TurnStatus.PENDING,
        "tool_calls_used": None,
        "attempted_tools": None,
        "replan_count": None,
    }


def attempted_tools(state: AgentState) -> set[ToolName]:
    return set(state.get("attempted_tools", []))


def untried_tools(state: AgentState, known: Iterable[ToolName]) -> list[ToolName]:
    """Tools this turn has not called yet. The fuel for a replan."""
    tried = attempted_tools(state)
    return [name for name in known if name not in tried]


def remaining_tool_budget(state: AgentState, limit: int) -> int:
    """Tool calls still allowed this turn.

    The budget is the circuit breaker on runaway orchestration: it caps the
    blast radius of a bad plan, a retry storm, or a future replan loop, so one
    turn can never spend unbounded money and latency.
    """
    return max(0, limit - int(state.get("tool_calls_used", 0)))


def trace_for_turn(state: AgentState, turn_index: int | None = None) -> list[TraceEvent]:
    """Slice the thread trace down to one turn, for CLI output and tests."""
    target = turn_index if turn_index is not None else state.get("turn_index", 0)
    return [
        event
        for event in state.get("trace", [])
        if event.attributes.get("turn_index") == target
    ]


def has_evidence(state: AgentState) -> bool:
    return bool(state.get("chunks") or state.get("web_results"))


def planned_tools(state: AgentState) -> list[ToolName]:
    plan = state.get("plan")
    return list(plan.tools) if plan else []
