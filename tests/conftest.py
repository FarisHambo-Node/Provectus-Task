from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

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


@pytest.fixture
def sqlite_path() -> Iterator[Path]:
    """A throwaway path for a SQLite checkpoint file.

    Deliberately not `tmp_path`: that fixture garbage-collects pytest's shared
    temp root, which fails hard if any leftover directory there is not
    deletable. This owns its directory and cleans up only that.
    """
    directory = Path(tempfile.mkdtemp(prefix="agent-checkpoints-"))
    try:
        yield directory / "checkpoints.sqlite"
    finally:
        shutil.rmtree(directory, ignore_errors=True)
