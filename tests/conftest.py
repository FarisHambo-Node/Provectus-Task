from __future__ import annotations

import pytest

from agent.container import AgentContainer, build_container
from agent.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """Deterministic, latency-free settings for tests."""
    return Settings(
        simulate_latency=False,
        simulated_failure_rate=0.0,
        random_seed=1234,
        tool_timeout_s=2.0,
        tool_max_attempts=2,
        tool_base_backoff_s=0.001,
        tool_max_backoff_s=0.002,
        log_level="CRITICAL",
    )


@pytest.fixture
def container(settings: Settings) -> AgentContainer:
    return build_container(settings)
