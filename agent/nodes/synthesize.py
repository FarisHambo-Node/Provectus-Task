"""Synthesis: turn evidence into one cited answer.

`build_citations` and `render_answer` are pure so the citation contract can be
tested without AWS. The node adds tracing, the post-check, and the AI message
that goes back into conversation state.

TODO: replace `render_answer` with a Bedrock Converse call. The prompt should
carry the numbered evidence block verbatim, instruct the model to answer only
from it, and require a `[n]` marker on every claim. Keep `render_answer` as the
deterministic fallback when generation fails — a cited extract beats an error.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from agent.container import AgentContainer
from agent.errors import SynthesisError
from agent.nodes._common import node_span
from agent.observability import TraceEvent
from agent.state import AgentState, Citation, RetrievedChunk, TurnStatus, WebResult

MAX_SNIPPET_CHARS = 280
# Internal docs are authoritative, so they get the low citation numbers.
INTERNAL_ORIGIN = "internal_docs"
WEB_ORIGIN = "web"


def build_citations(
    chunks: list[RetrievedChunk],
    web_results: list[WebResult],
) -> list[Citation]:
    """Number every piece of evidence once, internal sources first."""
    citations: list[Citation] = []
    seen: set[str] = set()

    for chunk in chunks:
        if chunk.source_uri in seen:
            continue
        seen.add(chunk.source_uri)
        citations.append(
            Citation(
                marker=f"[{len(citations) + 1}]",
                origin=INTERNAL_ORIGIN,
                title=chunk.title,
                locator=chunk.source_uri,
                snippet=_truncate(chunk.text),
                score=chunk.score,
            )
        )
    for result in web_results:
        if result.url in seen:
            continue
        seen.add(result.url)
        citations.append(
            Citation(
                marker=f"[{len(citations) + 1}]",
                origin=WEB_ORIGIN,
                title=result.title,
                locator=result.url,
                snippet=_truncate(result.snippet),
                score=result.score,
            )
        )
    return citations


def render_answer(
    query: str,
    citations: list[Citation],
    *,
    degraded_notes: list[str] | None = None,
) -> str:
    """Deterministic extractive answer with inline markers."""
    if not citations:
        raise SynthesisError("cannot synthesise without citations", context={"query": query})

    lines = [f"Answer to: {query}", ""]
    internal = [c for c in citations if c.origin == INTERNAL_ORIGIN]
    external = [c for c in citations if c.origin == WEB_ORIGIN]

    if internal:
        lines.append("From internal documentation:")
        lines += [f"- {c.snippet} {c.marker}" for c in internal]
        lines.append("")
    if external:
        lines.append("From external sources:")
        lines += [f"- {c.snippet} {c.marker}" for c in external]
        lines.append("")
    if degraded_notes:
        lines.append("Caveats:")
        lines += [f"- {note}" for note in degraded_notes]
        lines.append("")

    lines.append("Sources:")
    lines += [f"  {c.marker} {c.title} — {c.locator}" for c in citations]
    return "\n".join(lines).strip()


def verify_grounding(answer: str, citations: list[Citation]) -> list[str]:
    """Post-check: every answer must reference at least one real citation.

    Returns the problems found so the caller can decide to regenerate or
    downgrade the turn, rather than raising on a cosmetic issue.
    """
    problems: list[str] = []
    if not answer.strip():
        problems.append("empty answer")
    markers = {citation.marker for citation in citations}
    if markers and not any(marker in answer for marker in markers):
        problems.append("answer contains no citation marker")
    return problems


def degraded_notes(state: AgentState) -> list[str]:
    """Human-readable notes for tools that failed but did not sink the turn."""
    notes: list[str] = []
    for failure in state.get("failures", []):
        tool = failure.get("context", {}).get("tool") or failure.get("tool") or "a tool"
        notes.append(f"{tool} was unavailable ({failure.get('code')}), so its sources are missing.")
    return notes


async def synthesize(state: AgentState, container: AgentContainer) -> dict[str, Any]:
    trace: list[TraceEvent] = []
    async with node_span("synthesize", state, container, trace) as attrs:
        query = state.get("query", "")
        chunks = state.get("chunks", [])
        web_results = state.get("web_results", [])
        citations = build_citations(chunks, web_results)
        notes = degraded_notes(state)

        try:
            answer = render_answer(query, citations, degraded_notes=notes)
        except SynthesisError:
            # Routing should have prevented this; treat it as a real failure
            # rather than inventing an answer.
            container.logger.error("synthesis.no_citations", query_chars=len(query))
            raise

        problems = verify_grounding(answer, citations)
        attrs["citations"] = len(citations)
        attrs["answer_chars"] = len(answer)
        attrs["degraded"] = bool(notes)
        attrs["grounding_problems"] = problems

        if problems:
            container.metrics.increment("synthesis.grounding_problem")
            container.logger.warning("synthesis.grounding_problem", problems=problems)

        status = TurnStatus.DEGRADED if (notes or problems) else TurnStatus.OK
        return {
            "answer": answer,
            "citations": citations,
            "status": status,
            "messages": [AIMessage(content=answer)],
            "trace": trace,
        }


def _truncate(text: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"
