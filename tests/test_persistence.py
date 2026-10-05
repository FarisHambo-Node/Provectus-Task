"""Checkpointing: backend selection, durability, and degradation.

The SQLite tests write to a throwaway file and build a second agent against it,
which is the closest thing to a process restart that a test can do.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.container import build_container
from agent.errors import CheckpointError, ConfigurationError
from agent.graph import ConversationAgent
from agent.observability import MetricsSink, get_logger
from agent.persistence import (
    CheckpointBackend,
    CheckpointPolicy,
    ResilientCheckpointSaver,
    checkpointer_scope,
)
from agent.settings import Settings
from tests.test_graph import StubRetriever, StubWebBackend


def base_settings(**overrides) -> Settings:
    return Settings(
        simulate_latency=False,
        random_seed=99,
        log_level="CRITICAL",
        checkpoint_timeout_s=1.0,
        checkpoint_max_attempts=3,
        **overrides,
    )


def make_agent(settings: Settings, *, checkpointer=None) -> ConversationAgent:
    container = build_container(
        settings, retriever=StubRetriever(), web_backend=StubWebBackend()
    )
    return ConversationAgent(container, checkpointer=checkpointer)


class FlakySaver(InMemorySaver):
    """Fails the first `fail_times` calls to the named method, then behaves."""

    def __init__(self, *, method: str, fail_times: int, error: Exception | None = None):
        super().__init__()
        self.method = method
        self.remaining = fail_times
        self.error = error or ConnectionError("store unreachable")
        self.attempts = 0

    def _maybe_fail(self, method: str) -> None:
        if method != self.method:
            return
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error

    async def aget_tuple(self, config):
        self._maybe_fail("aget_tuple")
        return await super().aget_tuple(config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        self._maybe_fail("aput")
        return await super().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        self._maybe_fail("aput_writes")
        return await super().aput_writes(config, writes, task_id, task_path)


def wrap(inner, **policy_overrides) -> ResilientCheckpointSaver:
    return ResilientCheckpointSaver(
        inner,
        policy=CheckpointPolicy(timeout_s=1.0, max_attempts=3, **policy_overrides),
        logger=get_logger("test"),
        metrics=MetricsSink(),
    )


# --- backend selection -----------------------------------------------------


async def test_memory_backend_is_the_default():
    settings = base_settings()
    assert settings.checkpoint_backend == str(CheckpointBackend.MEMORY)
    async with checkpointer_scope(settings) as saver:
        assert isinstance(saver, ResilientCheckpointSaver)
        assert isinstance(saver.inner, InMemorySaver)


async def test_resilience_wrapper_can_be_disabled():
    async with checkpointer_scope(base_settings(), resilient=False) as saver:
        assert isinstance(saver, InMemorySaver)


async def test_unknown_backend_is_a_configuration_error():
    with pytest.raises(ConfigurationError) as exc:
        async with checkpointer_scope(base_settings(checkpoint_backend="redis")):
            pass
    assert "supported" in exc.value.context


async def test_sqlite_backend_creates_its_file(sqlite_path):
    path = sqlite_path.parent / "nested" / "checkpoints.sqlite"
    settings = base_settings(checkpoint_backend="sqlite", checkpoint_path=str(path))
    async with checkpointer_scope(settings) as saver:
        assert isinstance(saver, ResilientCheckpointSaver)
    assert path.exists()


# --- durability ------------------------------------------------------------


async def test_conversation_survives_a_new_agent_on_the_same_file(sqlite_path):
    settings = base_settings(
        checkpoint_backend="sqlite",
        checkpoint_path=str(sqlite_path),
    )
    async with ConversationAgent.session(
        build_container(settings, retriever=StubRetriever(), web_backend=StubWebBackend())
    ) as agent:
        first = await agent.ask("how do we chunk documents?", thread_id="t")
    assert first["turn_index"] == 1

    # New agent, new checkpointer connection: the same thread must continue.
    async with ConversationAgent.session(
        build_container(settings, retriever=StubRetriever(), web_backend=StubWebBackend())
    ) as agent:
        second = await agent.ask("how do we deploy the api?", thread_id="t")
        history = await agent.history(thread_id="t")

    assert second["turn_index"] == 2
    assert [entry["role"] for entry in history] == ["human", "ai", "human", "ai"]
    assert history[0]["content"] == "how do we chunk documents?"


async def test_in_memory_state_does_not_survive_a_new_agent():
    settings = base_settings()
    first = make_agent(settings)
    await first.ask("how do we chunk documents?", thread_id="t")
    second = make_agent(settings)
    assert (await second.state(thread_id="t"))["exists"] is False


# --- retry -----------------------------------------------------------------


async def test_transient_read_failure_is_retried():
    inner = FlakySaver(method="aget_tuple", fail_times=2)
    saver = wrap(inner)
    agent = make_agent(base_settings(), checkpointer=saver)

    result = await agent.ask("how do we chunk documents?", thread_id="t")
    assert result["status"] == "ok"
    assert inner.remaining == 0


async def test_transient_write_failure_is_retried():
    inner = FlakySaver(method="aput", fail_times=1)
    agent = make_agent(base_settings(), checkpointer=wrap(inner))

    result = await agent.ask("how do we chunk documents?", thread_id="t")
    assert result["status"] == "ok"
    # The turn persisted despite the first write failing.
    assert (await agent.state(thread_id="t"))["turn_index"] == 1


# --- degradation -----------------------------------------------------------


async def test_unreadable_store_degrades_to_a_stateless_turn():
    inner = FlakySaver(method="aget_tuple", fail_times=99)
    saver = wrap(inner, fail_open_reads=True)
    agent = make_agent(base_settings(), checkpointer=saver)

    first = await agent.ask("how do we chunk documents?", thread_id="t")
    second = await agent.ask("how do we deploy the api?", thread_id="t")

    # Both turns are answered, but neither can see the other.
    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert second["turn_index"] == 1


async def test_unwritable_store_still_answers_the_turn():
    inner = FlakySaver(method="aput", fail_times=99)
    saver = wrap(inner, fail_open_writes=True)
    agent = make_agent(base_settings(), checkpointer=saver)

    result = await agent.ask("how do we chunk documents?", thread_id="t")
    assert result["status"] == "ok"
    assert result["citations"]


async def test_failure_is_recorded_in_metrics_and_not_hidden():
    inner = FlakySaver(method="aget_tuple", fail_times=99)
    metrics = MetricsSink()
    saver = ResilientCheckpointSaver(
        inner,
        policy=CheckpointPolicy(timeout_s=1.0, max_attempts=1, fail_open_reads=True),
        logger=get_logger("test"),
        metrics=metrics,
    )
    agent = make_agent(base_settings(), checkpointer=saver)
    await agent.ask("how do we chunk documents?", thread_id="t")

    assert any("checkpoint.failure" in key for key in metrics.snapshot()["counters"])


async def test_fail_closed_reads_raise_a_typed_error():
    inner = FlakySaver(method="aget_tuple", fail_times=99)
    saver = wrap(inner, fail_open_reads=False)
    agent = make_agent(base_settings(), checkpointer=saver)

    with pytest.raises(CheckpointError) as exc:
        await agent.ask("how do we chunk documents?", thread_id="t")
    assert exc.value.context["operation"] == "read"
    assert exc.value.retryable is True


async def test_non_retryable_read_failure_fails_fast():
    inner = FlakySaver(
        method="aget_tuple", fail_times=99, error=ValueError("corrupt row")
    )
    saver = wrap(inner, fail_open_reads=True)
    agent = make_agent(base_settings(), checkpointer=saver)

    await agent.ask("how do we chunk documents?", thread_id="t")
    # One attempt, not three: retrying a corrupt row cannot help.
    assert inner.attempts == 1


# --- deletion --------------------------------------------------------------


async def test_deleting_a_thread_forgets_it(sqlite_path):
    settings = base_settings(
        checkpoint_backend="sqlite", checkpoint_path=str(sqlite_path)
    )
    async with ConversationAgent.session(
        build_container(settings, retriever=StubRetriever(), web_backend=StubWebBackend())
    ) as agent:
        await agent.ask("how do we chunk documents?", thread_id="doomed")
        await agent.ask("how do we chunk documents?", thread_id="kept")
        await agent.delete_thread(thread_id="doomed")

        assert (await agent.state(thread_id="doomed"))["exists"] is False
        assert (await agent.state(thread_id="kept"))["exists"] is True
