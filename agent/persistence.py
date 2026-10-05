"""Checkpointing: where conversation state lives between turns.

A checkpointer is what makes the agent multi-turn. On every superstep LangGraph
writes the channel values for a `thread_id`; on the next `ainvoke` it reloads
them, which is how `messages`, `turn_index`, and `summary` survive while the
per-turn channels are rebuilt.

Two pieces here:

* `checkpointer_scope` picks a backend from settings and owns its lifecycle.
* `ResilientCheckpointSaver` wraps *any* saver with timeout, retry, metrics,
  and an explicit degradation policy. A checkpoint store is a hard dependency
  on the request path, so when DynamoDB throttles or the disk fills, the
  behaviour has to be a decision rather than an accident.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

from agent.errors import ConfigurationError
from agent.observability import MetricsSink, StructuredLogger, get_logger
from agent.settings import Settings
from agent.tools.resilience import ResiliencePolicy, call_with_resilience


class CheckpointBackend(StrEnum):
    MEMORY = "memory"
    SQLITE = "sqlite"


@dataclass(frozen=True)
class CheckpointPolicy:
    """Resilience and degradation knobs for the checkpoint store."""

    timeout_s: float = 3.0
    max_attempts: int = 3
    # On read failure: start the turn with empty history instead of erroring.
    fail_open_reads: bool = True
    # On write failure: answer the turn but lose its persistence.
    fail_open_writes: bool = True

    @classmethod
    def from_settings(cls, settings: Settings) -> "CheckpointPolicy":
        return cls(
            timeout_s=settings.checkpoint_timeout_s,
            max_attempts=settings.checkpoint_max_attempts,
            fail_open_reads=settings.checkpoint_fail_open_reads,
            fail_open_writes=settings.checkpoint_fail_open_writes,
        )


@asynccontextmanager
async def checkpointer_scope(
    settings: Settings,
    *,
    logger: StructuredLogger | None = None,
    metrics: MetricsSink | None = None,
    resilient: bool = True,
) -> AsyncIterator[BaseCheckpointSaver]:
    """Open the configured checkpoint store for the lifetime of the block.

    SQLite needs its connection closed, so this is a context manager rather
    than a plain factory. In a long-lived service you open it once at startup
    and keep it for the process; do not open one per request.
    """
    log = logger or get_logger("agent.persistence")
    try:
        backend = CheckpointBackend(settings.checkpoint_backend)
    except ValueError as exc:
        raise ConfigurationError(
            "unknown checkpoint backend",
            context={
                "value": settings.checkpoint_backend,
                "supported": [str(b) for b in CheckpointBackend],
            },
        ) from exc

    def finish(saver: BaseCheckpointSaver) -> BaseCheckpointSaver:
        log.info("checkpointer.ready", backend=str(backend), resilient=resilient)
        if not resilient:
            return saver
        return ResilientCheckpointSaver(
            saver,
            policy=CheckpointPolicy.from_settings(settings),
            logger=log,
            metrics=metrics,
        )

    if backend is CheckpointBackend.MEMORY:
        # Dies with the process: fine for tests and a single-shot CLI, wrong
        # for anything with more than one replica.
        yield finish(InMemorySaver())
        return

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ConfigurationError(
            "sqlite checkpointing needs langgraph-checkpoint-sqlite",
            context={"install": "pip install langgraph-checkpoint-sqlite"},
        ) from exc

    path = Path(settings.checkpoint_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(
            "cannot create the checkpoint directory",
            context={"path": str(path), "error": repr(exc)},
        ) from exc

    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        log.info("checkpointer.opened", backend=str(backend), path=str(path))
        yield finish(saver)


class ResilientCheckpointSaver(BaseCheckpointSaver):
    """Timeout, retry, metrics, and a degradation policy around any saver.

    Reads and writes are retried because both are idempotent: a checkpoint is
    keyed by its id and a pending write by `(task_id, index)`, so replaying one
    cannot corrupt state.

    Only the async methods are hardened. The graph uses them for `ainvoke`; the
    sync methods delegate straight through so tooling that pokes at the store
    still works.
    """

    def __init__(
        self,
        inner: BaseCheckpointSaver,
        *,
        policy: CheckpointPolicy | None = None,
        logger: StructuredLogger | None = None,
        metrics: MetricsSink | None = None,
    ) -> None:
        super().__init__(serde=inner.serde)
        self.inner = inner
        self.policy = policy or CheckpointPolicy()
        self._log = logger or get_logger("agent.persistence")
        self._metrics = metrics
        self._rng = random.Random()
        self._retry = ResiliencePolicy(
            timeout_s=self.policy.timeout_s,
            max_attempts=self.policy.max_attempts,
            base_backoff_s=0.05,
            max_backoff_s=0.5,
            max_concurrency=1,
        )

    # -- async path, hardened ---------------------------------------------

    async def aget_tuple(self, config: Any) -> CheckpointTuple | None:
        try:
            result, _ = await self._call("read", lambda: self.inner.aget_tuple(config))
            return result
        except Exception as exc:
            self._degrade("read", exc, config=config, allowed=self.policy.fail_open_reads)
            # Returning None makes LangGraph treat this as a fresh thread: the
            # user keeps their answer but loses the conversation so far.
            return None

    async def aput(
        self,
        config: Any,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Any:
        try:
            result, _ = await self._call(
                "write",
                lambda: self.inner.aput(config, checkpoint, metadata, new_versions),
            )
            return result
        except Exception as exc:
            self._degrade("write", exc, config=config, allowed=self.policy.fail_open_writes)
            return _synthetic_config(config, checkpoint)

    async def aput_writes(
        self,
        config: Any,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        try:
            await self._call(
                "write_pending",
                lambda: self.inner.aput_writes(config, writes, task_id, task_path),
            )
        except Exception as exc:
            self._degrade(
                "write_pending", exc, config=config, allowed=self.policy.fail_open_writes
            )

    async def alist(
        self,
        config: Any | None,
        *,
        filter: dict[str, Any] | None = None,
        before: Any | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # Listing is a diagnostic path (history, time travel), not the request
        # path, so it surfaces errors instead of quietly returning nothing.
        async for item in self.inner.alist(config, filter=filter, before=before, limit=limit):
            yield item

    async def adelete_thread(self, thread_id: str) -> None:
        await self._call("delete", lambda: self.inner.adelete_thread(thread_id))
        self._log.info("checkpoint.thread_deleted", thread_id=thread_id)

    # -- sync path, passthrough -------------------------------------------

    def get_tuple(self, config: Any) -> CheckpointTuple | None:
        return self.inner.get_tuple(config)

    def put(
        self,
        config: Any,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Any:
        return self.inner.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: Any,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.inner.put_writes(config, writes, task_id, task_path)

    def list(
        self,
        config: Any | None,
        *,
        filter: dict[str, Any] | None = None,
        before: Any | None = None,
        limit: int | None = None,
    ):
        return self.inner.list(config, filter=filter, before=before, limit=limit)

    def delete_thread(self, thread_id: str) -> None:
        self.inner.delete_thread(thread_id)

    def get_next_version(self, current: Any, channel: Any = None) -> Any:
        return self.inner.get_next_version(current, channel)

    @property
    def config_specs(self) -> list[Any]:
        return self.inner.config_specs

    # -- internals ---------------------------------------------------------

    async def _call(self, operation: str, fn):
        return await call_with_resilience(
            fn,
            tool=f"checkpointer.{operation}",
            policy=self._retry,
            logger=self._log,
            metrics=self._metrics,
            rng=self._rng,
        )

    def _degrade(
        self,
        operation: str,
        exc: BaseException,
        *,
        config: Any,
        allowed: bool,
    ) -> None:
        """Record the failure, then either swallow it or let it through."""
        if isinstance(exc, asyncio.CancelledError):
            raise exc
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if self._metrics is not None:
            self._metrics.increment(
                "checkpoint.failure", 1.0, operation=operation, degraded=allowed
            )
        self._log.error(
            "checkpoint.failed",
            operation=operation,
            thread_id=thread_id,
            degraded=allowed,
            error=repr(exc),
        )
        if not allowed:
            raise exc


def _synthetic_config(config: Any, checkpoint: Checkpoint) -> dict[str, Any]:
    """The config `aput` would have returned, for the fail-open write path.

    The graph needs a well-formed config to keep running; the checkpoint it
    points at simply does not exist in the store.
    """
    configurable = (config or {}).get("configurable", {})
    return {
        "configurable": {
            "thread_id": configurable.get("thread_id"),
            "checkpoint_ns": configurable.get("checkpoint_ns", ""),
            "checkpoint_id": checkpoint["id"],
        }
    }
