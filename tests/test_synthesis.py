"""Citation numbering and the grounding post-check."""

from __future__ import annotations

import pytest

from agent.errors import SynthesisError
from agent.nodes.gather import classify_turn
from agent.nodes.synthesize import (
    build_citations,
    degraded_notes,
    render_answer,
    verify_grounding,
)
from agent.state import Citation, RetrievedChunk, ToolInvocation, TurnStatus, WebResult


def chunk(doc_id: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}#0",
        doc_id=doc_id,
        title=f"Doc {doc_id}",
        source_uri=f"s3://docs/{doc_id}.md",
        text="Default chunk size is 1000 characters with 150 of overlap.",
        score=score,
    )


def web(url: str, score: float = 0.5) -> WebResult:
    return WebResult(url=url, title=f"Page {url}", snippet="External snippet.", score=score)


# --- citations -------------------------------------------------------------


def test_citations_are_numbered_from_one_with_internal_sources_first():
    citations = build_citations([chunk("a"), chunk("b")], [web("https://x")])
    assert [c.marker for c in citations] == ["[1]", "[2]", "[3]"]
    assert [c.origin for c in citations] == ["internal_docs", "internal_docs", "web"]


def test_duplicate_sources_are_cited_once():
    citations = build_citations([chunk("a"), chunk("a")], [])
    assert len(citations) == 1


def test_no_evidence_produces_no_citations():
    assert build_citations([], []) == []


def test_long_snippets_are_truncated():
    long_chunk = chunk("a").model_copy(update={"text": "word " * 400})
    citation = build_citations([long_chunk], [])[0]
    assert len(citation.snippet) <= 280


# --- answer rendering ------------------------------------------------------


def test_answer_contains_every_marker_and_a_source_list():
    citations = build_citations([chunk("a")], [web("https://x")])
    answer = render_answer("how do we chunk?", citations)
    assert all(citation.marker in answer for citation in citations)
    assert "Sources:" in answer
    assert "s3://docs/a.md" in answer


def test_rendering_without_citations_is_an_error():
    # Refusing beats emitting an uncited answer; routing should prevent this.
    with pytest.raises(SynthesisError):
        render_answer("anything", [])


def test_caveats_are_surfaced_when_a_tool_failed():
    citations = build_citations([chunk("a")], [])
    answer = render_answer("q", citations, degraded_notes=["web_search was unavailable"])
    assert "Caveats:" in answer


# --- post-check ------------------------------------------------------------


def test_grounding_check_passes_on_a_cited_answer():
    citations = build_citations([chunk("a")], [])
    assert verify_grounding(render_answer("q", citations), citations) == []


def test_grounding_check_flags_a_missing_marker():
    citations = [
        Citation(marker="[1]", origin="internal_docs", title="t", locator="s3://a", snippet="s")
    ]
    assert verify_grounding("An answer with no marker.", citations) == [
        "answer contains no citation marker"
    ]


def test_grounding_check_flags_an_empty_answer():
    assert "empty answer" in verify_grounding("  ", [])


def test_degraded_notes_name_the_failed_tool():
    state = {"failures": [{"code": "tool_timeout", "context": {"tool": "web_search"}}]}
    notes = degraded_notes(state)
    assert len(notes) == 1
    assert "web_search" in notes[0]


# --- turn classification ---------------------------------------------------


def test_turn_with_evidence_and_no_failures_is_ok():
    state = {"chunks": [chunk("a")], "invocations": [ToolInvocation(tool="rag_search", ok=True)]}
    assert classify_turn(state) is TurnStatus.OK


def test_turn_with_evidence_and_a_failure_is_degraded():
    state = {
        "chunks": [chunk("a")],
        "invocations": [ToolInvocation(tool="rag_search", ok=True)],
        "failures": [{"code": "tool_timeout"}],
    }
    assert classify_turn(state) is TurnStatus.DEGRADED


def test_turn_where_every_tool_failed_is_failed():
    state = {
        "invocations": [ToolInvocation(tool="rag_search", ok=False)],
        "failures": [{"code": "tool_timeout"}],
    }
    assert classify_turn(state) is TurnStatus.FAILED


def test_turn_with_no_tool_calls_needs_clarification():
    assert classify_turn({}) is TurnStatus.NEEDS_CLARIFICATION
