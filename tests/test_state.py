"""Reducers decide how parallel branches merge, so they get tested directly."""

from __future__ import annotations

from agent.observability import TraceEvent
from agent.state import (
    MAX_TRACE_EVENTS,
    RetrievedChunk,
    WebResult,
    accumulate,
    accumulate_trace,
    dedupe_chunks,
    merge_chunks,
    merge_web_results,
)


def chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        title=f"title {chunk_id}",
        source_uri=f"s3://docs/{chunk_id}",
        text="body",
        score=score,
    )


def test_accumulate_appends():
    assert accumulate([1, 2], [3]) == [1, 2, 3]


def test_accumulate_treats_none_as_reset():
    assert accumulate([1, 2], None) == []


def test_accumulate_handles_missing_left():
    assert accumulate(None, [1]) == [1]


def test_merge_chunks_keeps_best_duplicate_and_ranks():
    merged = merge_chunks([chunk("a#0", 0.4)], [chunk("a#0", 0.9), chunk("b#0", 0.6)])
    assert [c.chunk_id for c in merged] == ["a#0", "b#0"]
    assert merged[0].score == 0.9


def test_merge_chunks_reset():
    assert merge_chunks([chunk("a#0", 0.4)], None) == []


def test_dedupe_chunks_is_stable_on_clean_input():
    chunks = [chunk("a#0", 0.9), chunk("b#0", 0.5)]
    assert dedupe_chunks(chunks) == chunks


def test_merge_web_results_dedupes_by_url():
    left = [WebResult(url="https://x", title="x", snippet="s", score=0.2)]
    right = [WebResult(url="https://x", title="x", snippet="s", score=0.8)]
    merged = merge_web_results(left, right)
    assert len(merged) == 1
    assert merged[0].score == 0.8


def test_trace_is_bounded():
    events = [TraceEvent(name=str(i), kind="node") for i in range(MAX_TRACE_EVENTS + 10)]
    merged = accumulate_trace([], events)
    assert len(merged) == MAX_TRACE_EVENTS
    # Oldest events are dropped, newest kept.
    assert merged[-1].name == str(MAX_TRACE_EVENTS + 9)
