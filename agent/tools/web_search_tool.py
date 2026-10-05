"""Web search tool: optional external evidence.

The backend is simulated. Results are synthesised from the query so a demo
answer is readable, and the same latency / failure injection as the retriever is
available for exercising the degraded path.

TODO: replace `SimulatedWebBackend` with a real provider client (Tavily, Brave,
Bing) behind the same `WebSearchBackend` protocol. Keep the SSRF and
allow/deny-list checks in the backend, not in the tool.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from agent.errors import ToolThrottledError, WebSearchError
from agent.observability import MetricsSink, StructuredLogger
from agent.state import ToolName, WebResult
from agent.tools.base import Tool, ToolRequest, ToolResult, ToolSpec
from agent.tools.resilience import ResiliencePolicy

WEB_SEARCH_TOOL_SPEC = ToolSpec(
    name=ToolName.WEB_SEARCH,
    description=(
        "Search the public web. Use only for current events, third-party "
        "releases, vendor pricing, or anything the internal docs cannot know."
    ),
    keywords=(
        "latest", "news", "today", "current", "recent", "release", "released",
        "announce", "announced", "changelog", "pricing", "price", "quota",
        "limits", "competitor", "benchmark", "versus", "vs", "2025", "2026",
        "state of the art", "upstream",
    ),
    is_default=False,
    cost_hint="medium",
)

MAX_RESULTS = 4


@runtime_checkable
class WebSearchBackend(Protocol):
    async def search(self, query: str, *, limit: int) -> list[WebResult]: ...


class SimulatedWebBackend:
    """Deterministic-ish fake provider with latency and failure injection."""

    _DOMAINS = (
        ("docs.aws.amazon.com", "AWS Documentation"),
        ("aws.amazon.com/blogs/machine-learning", "AWS ML Blog"),
        ("opensearch.org/docs", "OpenSearch Docs"),
        ("langchain-ai.github.io/langgraph", "LangGraph Docs"),
        ("arxiv.org", "arXiv preprint"),
    )

    def __init__(
        self,
        *,
        simulate_latency: bool = True,
        failure_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._simulate_latency = simulate_latency
        self._failure_rate = failure_rate
        self._rng = random.Random(seed)

    async def search(self, query: str, *, limit: int) -> list[WebResult]:
        if self._simulate_latency:
            await asyncio.sleep(self._rng.uniform(0.08, 0.3))
        self._maybe_fail()

        count = min(limit, self._rng.randint(2, MAX_RESULTS))
        picks = self._rng.sample(self._DOMAINS, min(count, len(self._DOMAINS)))
        topic = _topic(query)
        today = datetime.now(UTC).date().isoformat()
        return [
            WebResult(
                url=f"https://{domain}/search?q={topic.replace(' ', '+')}",
                title=f"{label}: {topic}",
                snippet=(
                    f"External source discussing {topic}. Simulated web result "
                    f"from {label}; swap in a real provider to get live text."
                ),
                score=round(self._rng.uniform(0.45, 0.95), 4),
                published_at=today,
            )
            for domain, label in picks
        ]

    def _maybe_fail(self) -> None:
        if self._failure_rate <= 0:
            return
        roll = self._rng.random()
        if roll >= self._failure_rate:
            return
        if roll < self._failure_rate / 2:
            raise ToolThrottledError("simulated provider 429", tool="web_search")
        raise WebSearchError("simulated provider 502", tool="web_search")


class WebSearchTool(Tool):
    """Fetch external results for queries the internal corpus cannot answer."""

    def __init__(
        self,
        backend: WebSearchBackend | None = None,
        *,
        policy: ResiliencePolicy | None = None,
        logger: StructuredLogger | None = None,
        metrics: MetricsSink | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(
            spec=WEB_SEARCH_TOOL_SPEC,
            policy=policy,
            logger=logger,
            metrics=metrics,
            rng=rng,
        )
        self._backend = backend or SimulatedWebBackend()

    async def _execute(self, request: ToolRequest) -> ToolResult:
        query = request.query.strip()
        if not query:
            raise WebSearchError("blank query reached the web tool", tool=str(self.name))

        results = await self._backend.search(query, limit=min(request.top_k, MAX_RESULTS))
        results.sort(key=lambda result: result.score, reverse=True)
        return ToolResult(tool=self.name, ok=True, web_results=results)


def _topic(query: str) -> str:
    """Trim a query down to something that reads like a headline."""
    words = [word for word in query.split() if word.lower() not in _STOPWORDS]
    return " ".join(words[:8]) or query[:60]


_STOPWORDS = frozenset(
    {"what", "whats", "what's", "is", "the", "a", "an", "of", "for", "to", "in", "on",
     "how", "do", "does", "did", "i", "we", "you", "about", "and", "or", "me", "tell"}
)
