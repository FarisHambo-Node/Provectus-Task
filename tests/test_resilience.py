"""Retry, backoff, and circuit breaker. These are the error-handling contract."""

from __future__ import annotations

import asyncio
import random

import pytest

from agent.errors import (
    CircuitOpenError,
    ToolPermanentError,
    ToolThrottledError,
    ToolTimeoutError,
)
from agent.observability import MetricsSink, get_logger
from agent.tools.resilience import (
    BreakerState,
    CircuitBreaker,
    ResiliencePolicy,
    backoff_delay,
    call_with_resilience,
)

POLICY = ResiliencePolicy(
    timeout_s=0.2, max_attempts=3, base_backoff_s=0.001, max_backoff_s=0.002
)


def runner(**kwargs):
    return dict(
        tool="rag_search",
        policy=kwargs.pop("policy", POLICY),
        logger=get_logger("test"),
        metrics=MetricsSink(),
        rng=random.Random(0),
        **kwargs,
    )


# --- backoff ---------------------------------------------------------------


def test_backoff_grows_and_is_capped():
    policy = ResiliencePolicy(base_backoff_s=0.5, max_backoff_s=1.0, jitter=False)
    rng = random.Random(0)
    assert backoff_delay(1, policy, rng) == 0.5
    assert backoff_delay(2, policy, rng) == 1.0
    assert backoff_delay(9, policy, rng) == 1.0


def test_jitter_stays_within_the_ceiling():
    policy = ResiliencePolicy(base_backoff_s=0.5, max_backoff_s=1.0, jitter=True)
    rng = random.Random(0)
    assert all(0.0 <= backoff_delay(2, policy, rng) <= 1.0 for _ in range(20))


# --- retries ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_call_runs_once():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        return "ok"

    result, attempts = await call_with_resilience(operation, **runner())
    assert (result, attempts, calls) == ("ok", 1, 1)


@pytest.mark.asyncio
async def test_retryable_error_is_retried_until_it_succeeds():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ToolThrottledError("429", tool="rag_search")
        return "ok"

    result, attempts = await call_with_resilience(operation, **runner())
    assert (result, attempts) == ("ok", 3)


@pytest.mark.asyncio
async def test_permanent_error_fails_fast():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise ToolPermanentError("bad request", tool="rag_search")

    with pytest.raises(ToolPermanentError):
        await call_with_resilience(operation, **runner())
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_are_exhausted_and_the_last_error_surfaces():
    async def operation():
        raise ToolThrottledError("429", tool="rag_search")

    with pytest.raises(ToolThrottledError):
        await call_with_resilience(operation, **runner())


@pytest.mark.asyncio
async def test_slow_call_times_out_as_a_tool_timeout():
    async def operation():
        await asyncio.sleep(1.0)

    with pytest.raises(ToolTimeoutError):
        await call_with_resilience(
            operation,
            **runner(policy=ResiliencePolicy(timeout_s=0.01, max_attempts=1)),
        )


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed():
    async def operation():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await call_with_resilience(operation, **runner())


# --- circuit breaker -------------------------------------------------------


def test_breaker_opens_after_threshold_and_half_opens_after_cooldown():
    now = [0.0]
    breaker = CircuitBreaker(
        name="rag_search", failure_threshold=2, reset_after_s=10.0, clock=lambda: now[0]
    )
    assert breaker.state is BreakerState.CLOSED

    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN

    with pytest.raises(CircuitOpenError):
        breaker.before_call()

    now[0] = 11.0
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.before_call()  # the probe is allowed through
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_success_resets_the_failure_count():
    breaker = CircuitBreaker(name="rag_search", failure_threshold=2)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED


@pytest.mark.asyncio
async def test_open_breaker_short_circuits_the_call():
    breaker = CircuitBreaker(name="rag_search", failure_threshold=1, reset_after_s=60.0)
    breaker.record_failure()
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(CircuitOpenError):
        await call_with_resilience(operation, **runner(breaker=breaker))
    assert calls == 0
