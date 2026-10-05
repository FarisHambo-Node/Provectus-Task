"""Query analysis: decide which tools this turn needs.

`plan_tools` is a pure function over (query, history, specs, settings) so the
routing policy can be unit tested without AWS. The node is a thin async wrapper
that adds tracing and error handling.

TODO: swap the heuristic for a Bedrock Converse call with
`registry.to_bedrock_tool_config()` and `toolChoice: {"auto": {}}`, then keep
`plan_tools` as the fallback when the model call fails or returns no tool use.
That ordering matters: the deterministic planner is what keeps the agent
answering during a Bedrock outage.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage

from agent.container import AgentContainer
from agent.errors import AgentError, PlanningError
from agent.memory import window
from agent.nodes._common import node_span
from agent.observability import TraceEvent
from agent.settings import Settings
from agent.state import AgentState, ToolName, ToolPlan, TurnStatus
from agent.tools.base import ToolSpec

MIN_QUERY_CHARS = 3
# Below this the planner is guessing; routing asks the user instead.
CLARIFICATION_CONFIDENCE = 0.2
FOLLOW_UP_MARKERS = (
    "that", "those", "it", "this", "they", "them", "the second", "the first",
    "same", "again", "instead", "what about", "and the",
)


def keyword_score(query: str, spec: ToolSpec) -> float:
    """Fraction of a tool's keywords present in the query, in [0, 1]."""
    if not spec.keywords:
        return 0.0
    lowered = query.lower()
    hits = sum(1 for keyword in spec.keywords if keyword in lowered)
    # Two matches is already a strong signal; don't require hitting every keyword.
    return min(1.0, hits / 2)


def looks_like_follow_up(query: str) -> bool:
    lowered = query.lower()
    return any(marker in f" {lowered} " for marker in FOLLOW_UP_MARKERS)


def resolve_query(query: str, history: list[AnyMessage], summary: str = "") -> str:
    """Make the query self-contained before it reaches a tool.

    Prefers the previous question from the live window, and falls back to the
    rolling summary once that turn has been compacted away — otherwise a
    follow-up in a long thread loses its referent and retrieval goes blind.

    TODO: real coreference resolution via a cheap LLM call over the last two
    turns; this keyword approach is the weakest part of the planner.
    """
    if not looks_like_follow_up(query):
        return query
    previous = [
        str(message.content).strip()
        for message in (history or [])[:-1]
        if getattr(message, "type", None) == "human"
    ]
    if previous:
        return f"{query} (context: {previous[-1]})"
    if summary.strip():
        return f"{query} (context: {_summary_tail(summary)})"
    return query


def _summary_tail(summary: str, max_chars: int = 240) -> str:
    """The most recent lines of the summary, which are the relevant ones."""
    tail = summary.strip().split("\n")[-2:]
    joined = " ".join(line.strip() for line in tail)
    return joined if len(joined) <= max_chars else joined[-max_chars:]


def plan_tools(
    query: str,
    *,
    specs: list[ToolSpec],
    settings: Settings,
    history: list[AnyMessage] | None = None,
    summary: str = "",
) -> ToolPlan:
    """Pick tools for a query. Pure, deterministic, no I/O."""
    cleaned = query.strip()
    if len(cleaned) < MIN_QUERY_CHARS:
        return ToolPlan(
            tools=[],
            rationale="query is too short to act on",
            confidence=0.0,
            needs_clarification=True,
            search_query=cleaned,
        )

    scored = sorted(
        ((spec, keyword_score(cleaned, spec)) for spec in specs),
        key=lambda pair: pair[1],
        reverse=True,
    )
    selected: list[ToolName] = [spec.name for spec, score in scored if score > 0.0]
    reasons = [f"{spec.name}={score:.2f}" for spec, score in scored]

    if not selected:
        # Unmatched questions are far more often about internal docs than the
        # web, so the default tool answers rather than asking for clarification.
        selected = [spec.name for spec in specs if spec.is_default]
        reasons.append("fell back to default tool")

    if not settings.enable_web_search:
        selected = [name for name in selected if name is not ToolName.WEB_SEARCH]

    selected = selected[: settings.max_tool_calls_per_turn]
    top_score = scored[0][1] if scored else 0.0
    confidence = round(min(1.0, 0.45 + 0.5 * top_score), 3) if selected else 0.0

    return ToolPlan(
        tools=selected,
        rationale="; ".join(reasons),
        confidence=confidence,
        needs_clarification=not selected,
        search_query=resolve_query(cleaned, history or [], summary),
    )


async def analyze_query(state: AgentState, container: AgentContainer) -> dict[str, Any]:
    trace: list[TraceEvent] = []
    async with node_span("analyze_query", state, container, trace) as attrs:
        query = state.get("query", "")
        summary = state.get("summary", "")
        attrs["has_summary"] = bool(summary)
        try:
            plan = plan_tools(
                query,
                specs=container.registry.specs(),
                settings=container.settings,
                history=window(
                    state.get("messages", []),
                    window_turns=container.settings.history_window_turns,
                ),
                summary=summary,
            )
        except AgentError:
            raise
        except Exception as exc:
            # A planner bug must not kill the turn: fall back to the default tool.
            container.logger.exception("planner.unhandled_error", query_chars=len(query))
            error = PlanningError("planner raised", context={"error": repr(exc)})
            plan = ToolPlan(
                tools=[
                    spec.name for spec in container.registry.specs() if spec.is_default
                ],
                rationale="planner failed, using default tool",
                confidence=0.3,
                search_query=query,
            )
            attrs["planner_error"] = error.code

        unknown = [name for name in plan.tools if not container.registry.has(name)]
        if unknown:
            # Guards against an LLM planner inventing tool names.
            plan = plan.model_copy(
                update={"tools": [n for n in plan.tools if container.registry.has(n)]}
            )
            attrs["dropped_unknown_tools"] = [str(name) for name in unknown]

        attrs["tools"] = [str(name) for name in plan.tools]
        attrs["confidence"] = plan.confidence
        attrs["needs_clarification"] = plan.needs_clarification

        update: dict[str, Any] = {"plan": plan, "trace": trace}
        if plan.needs_clarification or plan.confidence < CLARIFICATION_CONFIDENCE:
            update["status"] = TurnStatus.NEEDS_CLARIFICATION
        return update
