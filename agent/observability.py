"""Structured logging, trace spans, and in-process metrics.

Three pieces, all dependency-free so they work in a Lambda or an ECS task:

* `configure_logging` / `get_logger` emit one JSON object per line.
* `span` times a node or tool call and records a `TraceEvent` that travels in
  the graph state, so a finished turn carries its own trace.
* `MetricsSink` accumulates counters and latency histograms.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SpanKind = Literal["node", "tool", "retriever", "llm", "graph"]
SpanStatus = Literal["ok", "error", "skipped"]

# Set this attribute inside a span to mark the step as deliberately skipped.
SKIPPED_ATTR = "span.skipped"

_log_context: ContextVar[dict[str, Any]] = ContextVar("agent_log_context", default={})
_configured = False


class TraceEvent(BaseModel):
    """One timed step of a turn. Accumulated into `AgentState.trace`."""

    name: str
    kind: SpanKind
    status: SpanStatus = "ok"
    duration_ms: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None

    def summary(self) -> str:
        flag = {"ok": "ok", "error": "ERR", "skipped": "skip"}[self.status]
        return f"{self.name} [{flag}] {self.duration_ms:.0f}ms"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_log_context.get())
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class StructuredLogger:
    """Thin logging facade that takes keyword fields instead of f-strings."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def debug(self, msg: str, **fields: Any) -> None:
        self._log(logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._log(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._log(logging.WARNING, msg, fields)

    def error(self, msg: str, *, exc_info: bool = False, **fields: Any) -> None:
        self._log(logging.ERROR, msg, fields, exc_info=exc_info)

    def exception(self, msg: str, **fields: Any) -> None:
        self._log(logging.ERROR, msg, fields, exc_info=True)

    def _log(
        self,
        level: int,
        msg: str,
        fields: dict[str, Any],
        *,
        exc_info: bool = False,
    ) -> None:
        self._logger.log(level, msg, extra={"fields": fields}, exc_info=exc_info)


def configure_logging(level: str = "INFO", *, fmt: str = "json") -> None:
    """Install a single stderr handler. Safe to call more than once."""
    global _configured
    root = logging.getLogger()
    root.setLevel(level.upper())
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
        )
    root.handlers = [handler]
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach correlation fields (thread_id, turn, node) to every log in scope."""
    token = _log_context.set({**_log_context.get(), **fields})
    try:
        yield
    finally:
        _log_context.reset(token)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class MetricsSink:
    """Counters and latency samples for the current process.

    TODO: flush as CloudWatch EMF (one JSON line with an `_aws` block) or push
    to the OTLP collector sidecar instead of keeping samples in memory.
    """

    counters: dict[str, float] = field(default_factory=dict)
    latencies: dict[str, list[float]] = field(default_factory=dict)

    # Metric name and value are positional-only so that `**tags` can use any
    # key, including `name`.
    def increment(self, metric: str, value: float = 1.0, /, **tags: Any) -> None:
        key = self._key(metric, tags)
        self.counters[key] = self.counters.get(key, 0.0) + value

    def observe_latency(self, metric: str, duration_ms: float, /, **tags: Any) -> None:
        self.latencies.setdefault(self._key(metric, tags), []).append(duration_ms)

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": dict(sorted(self.counters.items())),
            "latency_ms": {
                key: {
                    "count": len(samples),
                    "p50": _percentile(samples, 0.50),
                    "p95": _percentile(samples, 0.95),
                    "max": max(samples),
                }
                for key, samples in sorted(self.latencies.items())
            },
        }

    @staticmethod
    def _key(metric: str, tags: dict[str, Any]) -> str:
        if not tags:
            return metric
        rendered = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric}{{{rendered}}}"


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return round(ordered[index], 2)


# --------------------------------------------------------------------------- #
# Spans
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def span(
    name: str,
    *,
    kind: SpanKind,
    logger: StructuredLogger,
    metrics: MetricsSink | None = None,
    trace: list[TraceEvent] | None = None,
    **attributes: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Time a step, emit logs/metrics, and append a `TraceEvent` to `trace`.

    The yielded dict is the span's attribute bag: mutate it inside the block to
    record outcome details (hit counts, scores, chosen tools).
    """
    from agent.errors import AgentError  # local import avoids a cycle

    attrs: dict[str, Any] = dict(attributes)
    event = TraceEvent(name=name, kind=kind, attributes=attrs)
    started = time.perf_counter()
    # Attributes are nested rather than splatted: a span is free to record a
    # key like `status` without colliding with the log record's own fields.
    logger.debug("span.start", span=name, kind=kind, attributes=attributes)
    try:
        yield attrs
    except BaseException as exc:
        event.status = "error"
        event.error = (
            exc.to_dict()
            if isinstance(exc, AgentError)
            else {"code": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        event.duration_ms = (time.perf_counter() - started) * 1000
        event.attributes = attrs
        if event.status == "ok" and attrs.get(SKIPPED_ATTR):
            event.status = "skipped"
        if trace is not None:
            trace.append(event)
        if metrics is not None:
            metrics.observe_latency(f"{kind}.duration", event.duration_ms, span=name)
            metrics.increment(f"{kind}.count", 1.0, span=name, status=event.status)
        log = logger.info if event.status != "error" else logger.error
        log(
            "span.end",
            span=name,
            kind=kind,
            status=event.status,
            duration_ms=round(event.duration_ms, 2),
            attributes=attrs,
        )
