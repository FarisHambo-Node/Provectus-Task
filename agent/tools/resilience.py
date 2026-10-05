"""Timeout, retry, and circuit-breaker primitives shared by every tool.

Kept out of the tool implementations so that adding a third tool cannot
accidentally ship a different retry policy.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from agent.errors import AgentError, CircuitOpenError, ToolTimeoutError, classify_exception
from agent.observability import MetricsSink, StructuredLogger

T = TypeVar("T")


@dataclass(frozen=True)
class ResiliencePolicy:
    """Per-tool knobs. Construct from `Settings` in the container."""

    timeout_s: float = 8.0
    max_attempts: int = 3
    base_backoff_s: float = 0.2
    max_backoff_s: float = 2.0
    # Full jitter: sleep is drawn from [0, computed_backoff].
    jitter: bool = True
    max_concurrency: int = 4
    breaker_failure_threshold: int = 4
    breaker_reset_after_s: float = 20.0


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Stops hammering a dependency that is already down.

    Opens after `failure_threshold` consecutive failures, then allows a single
    probe call once `reset_after_s` has elapsed.
    """

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int = 4,
        reset_after_s: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._reset_after = reset_after_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self._clock() - self._opened_at >= self._reset_after:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def before_call(self) -> None:
        if self.state is BreakerState.OPEN:
            raise CircuitOpenError(
                "circuit open, call not attempted",
                tool=self.name,
                context={
                    "consecutive_failures": self._failures,
                    "retry_after_s": round(
                        self._reset_after - (self._clock() - (self._opened_at or 0.0)), 2
                    ),
                },
            )

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._clock()


def backoff_delay(attempt: int, policy: ResiliencePolicy, rng: random.Random) -> float:
    """Exponential backoff with full jitter. `attempt` is 1-based."""
    ceiling = min(policy.max_backoff_s, policy.base_backoff_s * (2 ** (attempt - 1)))
    return rng.uniform(0.0, ceiling) if policy.jitter else ceiling


async def call_with_resilience(
    operation: Callable[[], Awaitable[T]],
    *,
    tool: str,
    policy: ResiliencePolicy,
    logger: StructuredLogger,
    metrics: MetricsSink | None = None,
    breaker: CircuitBreaker | None = None,
    semaphore: asyncio.Semaphore | None = None,
    rng: random.Random | None = None,
) -> tuple[T, int]:
    """Run `operation` under timeout + retry + breaker. Returns (result, attempts).

    Raises the classified `ToolError` from the final attempt. Only errors with
    `retryable=True` are retried; everything else fails fast so a malformed
    request does not burn the whole budget.
    """
    rng = rng or random.Random()
    last_error: AgentError | None = None

    for attempt in range(1, policy.max_attempts + 1):
        if breaker is not None:
            breaker.before_call()
        try:
            if semaphore is not None:
                async with semaphore:
                    result = await asyncio.wait_for(operation(), timeout=policy.timeout_s)
            else:
                result = await asyncio.wait_for(operation(), timeout=policy.timeout_s)
        except asyncio.CancelledError:
            # Cancellation is the caller going away, not a dependency failure.
            raise
        except TimeoutError as exc:
            last_error = ToolTimeoutError(
                f"tool exceeded {policy.timeout_s}s",
                tool=tool,
                context={"attempt": attempt},
            )
            _record_failure(last_error, tool, attempt, logger, metrics, breaker, exc=exc)
        except BaseException as exc:
            last_error = classify_exception(exc, tool=tool)
            _record_failure(last_error, tool, attempt, logger, metrics, breaker, exc=exc)
        else:
            if breaker is not None:
                breaker.record_success()
            if metrics is not None:
                metrics.increment("tool.success", tool=tool, attempts=attempt)
            return result, attempt

        if not last_error.retryable or attempt == policy.max_attempts:
            break
        delay = backoff_delay(attempt, policy, rng)
        logger.warning(
            "tool.retry_scheduled",
            tool=tool,
            attempt=attempt,
            max_attempts=policy.max_attempts,
            delay_s=round(delay, 3),
            error_code=last_error.code,
        )
        await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


def _record_failure(
    error: AgentError,
    tool: str,
    attempt: int,
    logger: StructuredLogger,
    metrics: MetricsSink | None,
    breaker: CircuitBreaker | None,
    *,
    exc: BaseException,
) -> None:
    if breaker is not None:
        breaker.record_failure()
    if metrics is not None:
        metrics.increment("tool.failure", tool=tool, code=error.code)
    logger.warning(
        "tool.attempt_failed",
        tool=tool,
        attempt=attempt,
        error_code=error.code,
        retryable=error.retryable,
        error=str(exc),
    )
