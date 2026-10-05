"""Dependency wiring.

One object built at startup holds settings, logger, metrics, the retriever, and
the tool registry. Nodes receive it by closure in `build_graph`, which keeps
them free of global state and trivially testable with fakes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from agent.observability import MetricsSink, StructuredLogger, configure_logging, get_logger
from agent.rag.retriever import Retriever, SimulatedVectorRetriever
from agent.settings import Settings
from agent.tools.base import Tool
from agent.tools.rag_tool import RagSearchTool
from agent.tools.registry import ToolRegistry
from agent.tools.resilience import ResiliencePolicy
from agent.tools.web_search_tool import SimulatedWebBackend, WebSearchBackend, WebSearchTool


@dataclass(frozen=True)
class AgentContainer:
    settings: Settings
    logger: StructuredLogger
    metrics: MetricsSink
    registry: ToolRegistry
    retriever: Retriever
    rng: random.Random = field(default_factory=random.Random)

    def resilience_policy(self) -> ResiliencePolicy:
        return _policy_from(self.settings)

    async def health_report(self) -> dict[str, bool]:
        """Probe every dependency. Call from the API's /health handler."""
        return {str(tool.name): await tool.health_check() for tool in self.registry}


def _policy_from(settings: Settings) -> ResiliencePolicy:
    return ResiliencePolicy(
        timeout_s=settings.tool_timeout_s,
        max_attempts=settings.tool_max_attempts,
        base_backoff_s=settings.tool_base_backoff_s,
        max_backoff_s=settings.tool_max_backoff_s,
        max_concurrency=settings.tool_max_concurrency,
        breaker_failure_threshold=settings.breaker_failure_threshold,
        breaker_reset_after_s=settings.breaker_reset_after_s,
    )


def build_container(
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
    web_backend: WebSearchBackend | None = None,
    extra_tools: tuple[Tool, ...] = (),
) -> AgentContainer:
    """Compose the agent. Pass fakes for `retriever`/`web_backend` in tests."""
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level, fmt=settings.log_format)
    logger = get_logger("agent")
    metrics = MetricsSink()
    rng = random.Random(settings.random_seed)
    policy = _policy_from(settings)

    retriever = retriever or SimulatedVectorRetriever(
        max_concurrency=settings.tool_max_concurrency,
        simulate_latency=settings.simulate_latency,
        failure_rate=settings.simulated_failure_rate,
        seed=settings.random_seed,
        logger=get_logger("agent.rag"),
        metrics=metrics,
    )

    tools: list[Tool] = [
        RagSearchTool(
            retriever,
            policy=policy,
            logger=get_logger("agent.tools.rag_search"),
            metrics=metrics,
            rng=rng,
        )
    ]
    if settings.enable_web_search:
        tools.append(
            WebSearchTool(
                web_backend
                or SimulatedWebBackend(
                    simulate_latency=settings.simulate_latency,
                    failure_rate=settings.simulated_failure_rate,
                    seed=settings.random_seed,
                ),
                policy=policy,
                logger=get_logger("agent.tools.web_search"),
                metrics=metrics,
                rng=rng,
            )
        )
    tools.extend(extra_tools)

    container = AgentContainer(
        settings=settings,
        logger=logger,
        metrics=metrics,
        registry=ToolRegistry(tools),
        retriever=retriever,
        rng=rng,
    )
    logger.info(
        "container.ready",
        tools=[str(name) for name in container.registry.names()],
        region=settings.aws_region,
        top_k=settings.top_k,
        web_search=settings.enable_web_search,
    )
    return container
