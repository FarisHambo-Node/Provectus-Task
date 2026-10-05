"""The RAG system: a `Retriever` protocol plus a simulated implementation.

This module knows nothing about tools, LangGraph, or conversations. It takes a
query string and returns scored chunks, which is exactly the contract the real
OpenSearch client will implement.

`SimulatedVectorRetriever` models the shape of the real thing rather than
faking the return value:

* an embedding call that blocks (so it goes through `asyncio.to_thread`),
* a concurrency semaphore, because every blocking call costs a thread,
* network latency,
* an injectable failure rate to exercise retry and fallback paths,
* `top_k` results drawn from the corpus with jittered similarity scores.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Protocol, runtime_checkable

from agent.errors import RetrievalError, ToolTimeoutError
from agent.observability import MetricsSink, StructuredLogger, get_logger
from agent.rag.corpus import SAMPLE_CORPUS, CorpusDocument
from agent.state import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a query into scored chunks."""

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        min_similarity: float = 0.0,
    ) -> list[RetrievedChunk]: ...

    async def health_check(self) -> bool: ...


# --------------------------------------------------------------------------- #
# Pure scoring helpers (unit-testable, no AWS, no I/O)
# --------------------------------------------------------------------------- #


def lexical_affinity(query: str, document: CorpusDocument) -> float:
    """Crude keyword overlap in [0, 1], standing in for cosine similarity."""
    terms = {term for term in _tokenize(query) if len(term) > 2}
    if not terms:
        return 0.0
    haystack = " ".join(
        (document.title, document.section, document.text, *document.keywords)
    ).lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def simulate_similarity(
    affinity: float,
    rng: random.Random,
    *,
    floor: float = 0.18,
    ceiling: float = 0.97,
) -> float:
    """Blend keyword affinity with noise so scores look like a real k-NN hit list."""
    noise = rng.uniform(-0.12, 0.12)
    score = 0.35 + 0.55 * affinity + noise
    return round(min(ceiling, max(floor, score)), 4)


def rank_candidates(
    query: str,
    documents: tuple[CorpusDocument, ...],
    *,
    top_k: int,
    rng: random.Random,
    sample_size: int | None = None,
) -> list[tuple[CorpusDocument, float]]:
    """Sample a candidate set, score it, and return the best `top_k`.

    Sampling first is what makes runs non-deterministic: an ANN index does not
    return the same neighbours every time either.
    """
    if top_k <= 0:
        return []
    pool = list(documents)
    size = min(len(pool), sample_size or max(top_k * 2, top_k + 5))
    candidates = rng.sample(pool, size)
    scored = [(doc, simulate_similarity(lexical_affinity(query, doc), rng)) for doc in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def to_chunk(document: CorpusDocument, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{document.doc_id}#0",
        doc_id=document.doc_id,
        title=document.title,
        source_uri=document.source_uri,
        text=document.text,
        score=score,
        section=document.section,
    )


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in "".join(c.lower() if c.isalnum() else " " for c in text).split()
        if token
    ]


# --------------------------------------------------------------------------- #
# Simulated retriever
# --------------------------------------------------------------------------- #


class SimulatedVectorRetriever:
    """Stand-in for OpenSearch k-NN over the Titan-embedded corpus."""

    def __init__(
        self,
        *,
        documents: tuple[CorpusDocument, ...] = SAMPLE_CORPUS,
        max_concurrency: int = 4,
        simulate_latency: bool = True,
        failure_rate: float = 0.0,
        seed: int | None = None,
        logger: StructuredLogger | None = None,
        metrics: MetricsSink | None = None,
    ) -> None:
        self._documents = documents
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._simulate_latency = simulate_latency
        self._failure_rate = failure_rate
        self._rng = random.Random(seed)
        self._log = logger or get_logger(__name__)
        self._metrics = metrics

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        min_similarity: float = 0.0,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise RetrievalError("empty query", tool="rag_search")

        started = time.perf_counter()
        async with self._semaphore:
            # The real path is `await asyncio.to_thread(bedrock.invoke_model, ...)`.
            vector = await asyncio.to_thread(self._embed_blocking, query)
            await self._simulate_network()
            self._maybe_fail(query)
            scored = rank_candidates(query, self._documents, top_k=top_k, rng=self._rng)

        chunks = [to_chunk(doc, score) for doc, score in scored if score >= min_similarity]
        duration_ms = (time.perf_counter() - started) * 1000
        if self._metrics is not None:
            self._metrics.observe_latency("retriever.duration", duration_ms)
            self._metrics.increment("retriever.chunks", len(chunks))
        self._log.info(
            "retriever.search",
            query_chars=len(query),
            embedding_dims=len(vector),
            candidates=len(scored),
            returned=len(chunks),
            dropped_below_floor=len(scored) - len(chunks),
            top_score=chunks[0].score if chunks else None,
            duration_ms=round(duration_ms, 2),
        )
        return chunks

    async def health_check(self) -> bool:
        """Cheap readiness probe. Real version hits the OpenSearch _cluster/health."""
        return bool(self._documents)

    def _embed_blocking(self, query: str) -> list[float]:
        """Deterministic pseudo-embedding; replaced by Titan Text Embeddings v2."""
        rng = random.Random(hash(query) & 0xFFFFFFFF)
        return [rng.uniform(-1.0, 1.0) for _ in range(8)]

    async def _simulate_network(self) -> None:
        if self._simulate_latency:
            await asyncio.sleep(self._rng.uniform(0.04, 0.18))

    def _maybe_fail(self, query: str) -> None:
        """Inject the failure modes the graph has to survive."""
        if self._failure_rate <= 0:
            return
        roll = self._rng.random()
        if roll >= self._failure_rate:
            return
        if roll < self._failure_rate / 2:
            raise ToolTimeoutError(
                "simulated OpenSearch timeout",
                tool="rag_search",
                context={"query_chars": len(query)},
            )
        raise RetrievalError(
            "simulated OpenSearch 503",
            tool="rag_search",
            context={"query_chars": len(query)},
        )
