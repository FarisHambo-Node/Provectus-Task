"""RAG tool: the adapter between the graph and the document store.

The tool owns the query-side concerns (validation, top_k, similarity floor,
telemetry). It does not know how retrieval works — that lives behind the
`Retriever` protocol, so swapping `SimulatedVectorRetriever` for the real
OpenSearch client touches only the container.
"""

from __future__ import annotations

import random

from agent.errors import RetrievalError
from agent.observability import MetricsSink, StructuredLogger
from agent.rag.retriever import Retriever
from agent.state import ToolName
from agent.tools.base import Tool, ToolRequest, ToolResult, ToolSpec
from agent.tools.resilience import ResiliencePolicy

RAG_TOOL_SPEC = ToolSpec(
    name=ToolName.RAG_SEARCH,
    description=(
        "Search the internal engineering and policy documentation. Use for "
        "anything about our own architecture, defaults, runbooks, or decisions."
    ),
    keywords=(
        "internal", "our", "we", "docs", "documentation", "runbook", "policy",
        "architecture", "config", "configuration", "default", "pipeline",
        "index", "embedding", "chunk", "retrieval", "opensearch", "bedrock",
        "deploy", "cost", "how do we", "where is",
    ),
    is_default=True,
    cost_hint="low",
)


class RagSearchTool(Tool):
    """Retrieve grounding chunks from the internal corpus."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        policy: ResiliencePolicy | None = None,
        logger: StructuredLogger | None = None,
        metrics: MetricsSink | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(
            spec=RAG_TOOL_SPEC, policy=policy, logger=logger, metrics=metrics, rng=rng
        )
        self._retriever = retriever

    async def _execute(self, request: ToolRequest) -> ToolResult:
        query = request.query.strip()
        if not query:
            raise RetrievalError("blank query reached the rag tool", tool=str(self.name))

        chunks = await self._retriever.search(
            query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )
        return ToolResult(tool=self.name, ok=True, chunks=chunks)

    async def health_check(self) -> bool:
        return await self._retriever.health_check()
