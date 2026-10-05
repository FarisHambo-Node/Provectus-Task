"""Graph-level behaviour: routing, multi-turn state, and graceful degradation.

Dependencies are faked, so these run offline and deterministically.
"""

from __future__ import annotations

import pytest

from agent.container import build_container
from agent.errors import RetrievalError, ThreadNotFoundError, ToolThrottledError
from agent.graph import ConversationAgent
from agent.nodes.tool_nodes import rag_search_node
from agent.rag.corpus import SAMPLE_CORPUS
from agent.rag.retriever import to_chunk
from agent.settings import Settings
from agent.state import RetrievedChunk, WebResult


class StubRetriever:
    """Returns a fixed slice of the corpus, or raises a chosen error."""

    def __init__(self, *, count: int = 3, error: Exception | None = None) -> None:
        self.count = count
        self.error = error
        self.calls: list[str] = []

    async def search(
        self, query: str, *, top_k: int, min_similarity: float = 0.0
    ) -> list[RetrievedChunk]:
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return [to_chunk(doc, 0.9 - i * 0.1) for i, doc in enumerate(SAMPLE_CORPUS[: self.count])]

    async def health_check(self) -> bool:
        return self.error is None


class StubWebBackend:
    def __init__(self, *, count: int = 2, error: Exception | None = None) -> None:
        self.count = count
        self.error = error
        self.calls: list[str] = []

    async def search(self, query: str, *, limit: int) -> list[WebResult]:
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return [
            WebResult(url=f"https://example.com/{i}", title=f"Result {i}", snippet="snippet", score=0.7)
            for i in range(self.count)
        ]


def make_agent(
    *,
    retriever: StubRetriever | None = None,
    web_backend: StubWebBackend | None = None,
    **overrides,
) -> ConversationAgent:
    settings = Settings(
        simulate_latency=False,
        random_seed=99,
        tool_max_attempts=2,
        tool_base_backoff_s=0.001,
        tool_max_backoff_s=0.002,
        log_level="CRITICAL",
        **overrides,
    )
    container = build_container(
        settings,
        retriever=retriever or StubRetriever(),
        web_backend=web_backend or StubWebBackend(),
    )
    return ConversationAgent(container)


# --- happy paths -----------------------------------------------------------


@pytest.mark.asyncio
async def test_internal_question_uses_rag_only_and_cites_it():
    web = StubWebBackend()
    agent = make_agent(web_backend=web)
    result = await agent.ask("where is our chunking configuration documented?")

    assert result["status"] == "ok"
    assert result["tools_used"] == ["rag_search"]
    assert web.calls == []
    assert result["citations"]
    assert all(c["marker"] in result["answer"] for c in result["citations"])


@pytest.mark.asyncio
async def test_mixed_question_runs_both_tools_and_merges_citations():
    retriever, web = StubRetriever(), StubWebBackend()
    agent = make_agent(retriever=retriever, web_backend=web)
    result = await agent.ask("how do we chunk docs and what is the latest OpenSearch release?")

    assert set(result["tools_used"]) == {"rag_search", "web_search"}
    assert len(retriever.calls) == 1
    assert len(web.calls) == 1
    origins = {c["origin"] for c in result["citations"]}
    assert origins == {"internal_docs", "web"}


# --- degraded and refusal paths -------------------------------------------


@pytest.mark.asyncio
async def test_turn_is_degraded_when_one_tool_fails_but_the_other_succeeds():
    agent = make_agent(
        web_backend=StubWebBackend(error=ToolThrottledError("429", tool="web_search"))
    )
    result = await agent.ask("how do we chunk docs and what is the latest release?")

    assert result["status"] == "degraded"
    assert result["tools_used"] == ["rag_search"]
    assert result["failures"]
    assert "Caveats:" in result["answer"]


@pytest.mark.asyncio
async def test_turn_refuses_when_every_tool_fails():
    agent = make_agent(retriever=StubRetriever(error=RetrievalError("503", tool="rag_search")))
    result = await agent.ask("where is our chunking configuration documented?")

    assert result["status"] == "failed"
    assert result["citations"] == []
    assert "not going to guess" in result["answer"]


@pytest.mark.asyncio
async def test_empty_retrieval_refuses_instead_of_inventing_an_answer():
    agent = make_agent(retriever=StubRetriever(count=0))
    result = await agent.ask("where is our chunking configuration documented?")

    assert result["status"] == "failed"
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_unanswerable_greeting_asks_for_clarification():
    retriever = StubRetriever()
    agent = make_agent(retriever=retriever)
    result = await agent.ask("hi")

    assert result["status"] == "needs_clarification"
    assert retriever.calls == []


# --- multi-turn state ------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_index_increments_and_history_accumulates():
    agent = make_agent()
    first = await agent.ask("how do we chunk documents?", thread_id="t1")
    second = await agent.ask("and how do we embed them?", thread_id="t1")

    assert (first["turn_index"], second["turn_index"]) == (1, 2)
    history = await agent.history(thread_id="t1")
    assert [entry["role"] for entry in history] == ["human", "ai", "human", "ai"]


@pytest.mark.asyncio
async def test_threads_are_isolated():
    agent = make_agent()
    await agent.ask("how do we chunk documents?", thread_id="a")
    other = await agent.ask("how do we chunk documents?", thread_id="b")

    assert other["turn_index"] == 1
    assert len(await agent.history(thread_id="b")) == 2


@pytest.mark.asyncio
async def test_scratch_state_does_not_leak_between_turns():
    agent = make_agent(retriever=StubRetriever(count=3))
    await agent.ask("how do we chunk documents?", thread_id="t2")
    second = await agent.ask("how do we deploy the api?", thread_id="t2")

    # Three chunks per call: without the per-turn reset this would be six.
    internal = [c for c in second["citations"] if c["origin"] == "internal_docs"]
    assert len(internal) == 3


@pytest.mark.asyncio
async def test_follow_up_query_is_rewritten_before_retrieval():
    retriever = StubRetriever()
    agent = make_agent(retriever=retriever)
    await agent.ask("how do we embed documents?", thread_id="t3")
    await agent.ask("what about that in production?", thread_id="t3")

    assert "how do we embed documents?" in retriever.calls[-1]


@pytest.mark.asyncio
async def test_a_failed_turn_does_not_break_the_next_one():
    retriever = StubRetriever(error=RetrievalError("503", tool="rag_search"))
    agent = make_agent(retriever=retriever)
    failed = await agent.ask("where is our chunking doc?", thread_id="t4")
    retriever.error = None
    recovered = await agent.ask("where is our chunking doc?", thread_id="t4")

    assert failed["status"] == "failed"
    assert recovered["status"] == "ok"
    assert recovered["failures"] == []


# --- observability ---------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_covers_the_executed_path_only():
    agent = make_agent()
    result = await agent.ask("where is our chunking configuration documented?")
    names = [entry.split(" ")[0] for entry in result["trace"]]

    assert "node.prepare_turn" in names
    assert "node.rag_search" in names
    assert "node.synthesize" in names
    assert "node.web_search" not in names


@pytest.mark.asyncio
async def test_metrics_record_tool_outcomes():
    agent = make_agent()
    await agent.ask("where is our chunking configuration documented?")
    snapshot = agent.container.metrics.snapshot()

    assert any("tool.success" in key for key in snapshot["counters"])
    assert any("node.duration" in key for key in snapshot["latency_ms"])


@pytest.mark.asyncio
async def test_health_report_covers_every_registered_tool():
    agent = make_agent()
    assert set(await agent.container.health_report()) == {"rag_search", "web_search"}


# --- tool-call budget (circuit breaker) ------------------------------------


@pytest.mark.asyncio
async def test_budget_counts_every_call_including_failures():
    agent = make_agent(
        web_backend=StubWebBackend(error=ToolThrottledError("429", tool="web_search"))
    )
    result = await agent.ask("how do we chunk docs and what is the latest release?")
    # Both tools ran; the failing one still spent its call.
    assert result["tool_calls_used"] == 2


@pytest.mark.asyncio
async def test_budget_of_one_allows_a_single_tool():
    web = StubWebBackend()
    agent = make_agent(web_backend=web, max_tool_calls_per_turn=1)
    result = await agent.ask("how do we chunk docs and what is the latest release?")

    assert result["tool_calls_used"] == 1
    assert web.calls == []
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_budget_resets_between_turns():
    agent = make_agent()
    await agent.ask("how do we chunk docs and what is the latest release?", thread_id="t")
    second = await agent.ask(
        "how do we chunk docs and what is the latest release?", thread_id="t"
    )
    # Without the per-turn reset this would be 4.
    assert second["tool_calls_used"] == 2


@pytest.mark.asyncio
async def test_tool_node_refuses_to_run_once_the_budget_is_spent():
    """The second line of defence, below the routing-level clip.

    Unreachable through the current single-pass graph because `prepare_turn`
    resets the counter and routing clips the fan-out. It exists so a replan
    loop cannot bypass the breaker, so it is tested at the node directly.
    """
    retriever = StubRetriever()
    container = build_container(
        Settings(simulate_latency=False, log_level="CRITICAL", max_tool_calls_per_turn=2),
        retriever=retriever,
        web_backend=StubWebBackend(),
    )
    state = {
        "query": "where is our chunking doc?",
        "thread_id": "t",
        "turn_index": 1,
        "tool_calls_used": 2,
    }

    update = await rag_search_node(state, container)

    assert retriever.calls == []
    assert update["failures"][0]["code"] == "budget_exceeded"
    assert "tool_calls_used" not in update


# --- memory compaction -----------------------------------------------------


def compacting_agent() -> ConversationAgent:
    """Summarise aggressively so a test does not need a 9-turn conversation."""
    return make_agent(history_window_turns=1, summarize_after_turns=2)


@pytest.mark.asyncio
async def test_summary_is_empty_until_the_threshold_is_crossed():
    agent = compacting_agent()
    first = await agent.ask("how do we chunk documents?", thread_id="t")
    assert first["summary"] == ""


@pytest.mark.asyncio
async def test_long_thread_is_summarised_and_the_checkpoint_shrinks():
    agent = compacting_agent()
    for question in (
        "how do we chunk documents?",
        "how do we embed them?",
        "how do we deploy the api?",
    ):
        result = await agent.ask(question, thread_id="t")

    state = await agent.state(thread_id="t")
    assert result["summary"]
    assert "how do we chunk documents?" in state["summary"]
    # Window of one turn: the stored transcript stops growing with the thread.
    assert state["stored_turns"] == 1
    assert state["stored_messages"] == 2


@pytest.mark.asyncio
async def test_summary_respects_the_character_budget():
    agent = make_agent(
        history_window_turns=1, summarize_after_turns=2, summary_max_chars=300
    )
    for i in range(5):
        await agent.ask(f"how do we handle case {i} in our pipeline?", thread_id="t")
    assert len((await agent.state(thread_id="t"))["summary"]) <= 300


@pytest.mark.asyncio
async def test_follow_up_survives_compaction_via_the_summary():
    retriever = StubRetriever()
    agent = make_agent(
        retriever=retriever, history_window_turns=1, summarize_after_turns=2
    )
    await agent.ask("how do we embed documents?", thread_id="t")
    await agent.ask("how do we deploy the api?", thread_id="t")
    await agent.ask("what about that?", thread_id="t")

    # The referenced turn is gone from the window, so the rewriter falls back
    # to the summary rather than sending a bare pronoun to retrieval.
    assert "context:" in retriever.calls[-1]


@pytest.mark.asyncio
async def test_summarization_node_only_runs_when_needed():
    agent = compacting_agent()
    first = await agent.ask("how do we chunk documents?", thread_id="t")
    assert not any("summarize_history" in entry for entry in first["trace"])

    await agent.ask("how do we embed them?", thread_id="t")
    third = await agent.ask("how do we deploy the api?", thread_id="t")
    assert any("summarize_history" in entry for entry in third["trace"])


# --- checkpoint inspection and time travel ---------------------------------


@pytest.mark.asyncio
async def test_state_reports_what_the_agent_remembers():
    agent = make_agent()
    await agent.ask("how do we chunk documents?", thread_id="t")
    state = await agent.state(thread_id="t")

    assert state["exists"] is True
    assert state["turn_index"] == 1
    assert state["stored_turns"] == 1
    assert state["checkpoint_id"]
    # The turn finished, so nothing is left to run.
    assert state["next_nodes"] == []


@pytest.mark.asyncio
async def test_state_of_an_unknown_thread_is_empty():
    agent = make_agent()
    assert (await agent.state(thread_id="never-used"))["exists"] is False


@pytest.mark.asyncio
async def test_checkpoints_are_listed_newest_first():
    agent = make_agent()
    await agent.ask("how do we chunk documents?", thread_id="t")
    await agent.ask("how do we deploy the api?", thread_id="t")

    completed = [c for c in await agent.checkpoints(thread_id="t", limit=60) if not c["next_nodes"]]
    assert [c["turn_index"] for c in completed] == [2, 1]
    assert all(c["checkpoint_id"] for c in completed)


async def completed_checkpoint(agent: ConversationAgent, thread_id: str, turn: int) -> str:
    entries = await agent.checkpoints(thread_id=thread_id, limit=60)
    return next(
        c["checkpoint_id"]
        for c in entries
        if c["turn_index"] == turn and not c["next_nodes"]
    )


@pytest.mark.asyncio
async def test_forking_in_place_replaces_the_last_question():
    agent = make_agent()
    await agent.ask("how do we chunk documents?", thread_id="t")
    await agent.ask("how do we deploy the api?", thread_id="t")
    after_first = await completed_checkpoint(agent, "t", 1)

    forked = await agent.fork(
        "what about PII redaction?", thread_id="t", checkpoint_id=after_first
    )
    questions = [
        entry["content"]
        for entry in await agent.history(thread_id="t")
        if entry["role"] == "human"
    ]

    assert forked["turn_index"] == 2
    assert questions == ["how do we chunk documents?", "what about PII redaction?"]


@pytest.mark.asyncio
async def test_forking_into_a_new_thread_leaves_the_original_alone():
    agent = make_agent()
    await agent.ask("how do we chunk documents?", thread_id="t")
    await agent.ask("how do we deploy the api?", thread_id="t")
    after_first = await completed_checkpoint(agent, "t", 1)

    await agent.fork(
        "how do we handle retries?",
        thread_id="t",
        checkpoint_id=after_first,
        into_thread_id="branch",
    )

    branch = [
        entry["content"]
        for entry in await agent.history(thread_id="branch")
        if entry["role"] == "human"
    ]
    original = [
        entry["content"]
        for entry in await agent.history(thread_id="t")
        if entry["role"] == "human"
    ]
    assert branch == ["how do we chunk documents?", "how do we handle retries?"]
    assert original == ["how do we chunk documents?", "how do we deploy the api?"]


@pytest.mark.asyncio
async def test_forking_from_an_unknown_checkpoint_is_an_error():
    agent = make_agent()
    await agent.ask("how do we chunk documents?", thread_id="t")
    with pytest.raises(ThreadNotFoundError):
        await agent.fork(
            "anything", thread_id="t", checkpoint_id="nope", into_thread_id="branch"
        )
