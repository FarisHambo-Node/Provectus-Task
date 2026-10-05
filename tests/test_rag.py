"""The simulated RAG system: scoring is pure, search is async but AWS-free."""

from __future__ import annotations

import random

import pytest

from agent.errors import RetrievalError, ToolError
from agent.rag.corpus import SAMPLE_CORPUS, corpus_size
from agent.rag.retriever import (
    SimulatedVectorRetriever,
    lexical_affinity,
    rank_candidates,
    simulate_similarity,
)


def test_corpus_has_twenty_unique_documents():
    assert corpus_size() == 20
    assert len({doc.doc_id for doc in SAMPLE_CORPUS}) == 20


def test_lexical_affinity_is_bounded_and_ordered():
    chunking_doc = next(doc for doc in SAMPLE_CORPUS if doc.doc_id == "doc-002")
    pii_doc = next(doc for doc in SAMPLE_CORPUS if doc.doc_id == "doc-019")
    query = "what is the default chunk size and overlap"
    assert 0.0 <= lexical_affinity(query, chunking_doc) <= 1.0
    assert lexical_affinity(query, chunking_doc) > lexical_affinity(query, pii_doc)


def test_lexical_affinity_of_empty_query_is_zero():
    assert lexical_affinity("", SAMPLE_CORPUS[0]) == 0.0


def test_simulated_similarity_stays_in_range():
    rng = random.Random(0)
    scores = [simulate_similarity(affinity, rng) for affinity in (0.0, 0.5, 1.0)]
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_rank_candidates_returns_top_k_sorted():
    rng = random.Random(42)
    ranked = rank_candidates("chunking defaults", SAMPLE_CORPUS, top_k=5, rng=rng)
    assert len(ranked) == 5
    assert [score for _, score in ranked] == sorted(
        (score for _, score in ranked), reverse=True
    )
    assert len({doc.doc_id for doc, _ in ranked}) == 5


def test_rank_candidates_with_zero_top_k():
    assert rank_candidates("x", SAMPLE_CORPUS, top_k=0, rng=random.Random(0)) == []


def test_same_seed_reproduces_the_same_ranking():
    args = ("hybrid search", SAMPLE_CORPUS)
    first = rank_candidates(*args, top_k=5, rng=random.Random(7))
    second = rank_candidates(*args, top_k=5, rng=random.Random(7))
    assert [d.doc_id for d, _ in first] == [d.doc_id for d, _ in second]


@pytest.mark.asyncio
async def test_search_returns_requested_number_of_chunks():
    retriever = SimulatedVectorRetriever(simulate_latency=False, seed=11)
    chunks = await retriever.search("how do we deploy the api", top_k=5)
    assert len(chunks) == 5
    assert all(chunk.source_uri for chunk in chunks)


@pytest.mark.asyncio
async def test_similarity_floor_drops_weak_chunks():
    retriever = SimulatedVectorRetriever(simulate_latency=False, seed=11)
    chunks = await retriever.search("anything", top_k=5, min_similarity=0.99)
    assert chunks == []


@pytest.mark.asyncio
async def test_blank_query_is_rejected():
    retriever = SimulatedVectorRetriever(simulate_latency=False)
    with pytest.raises(RetrievalError):
        await retriever.search("   ", top_k=5)


@pytest.mark.asyncio
async def test_failure_injection_raises():
    retriever = SimulatedVectorRetriever(
        simulate_latency=False, failure_rate=1.0, seed=5
    )
    # Either a timeout or a 503, depending on the roll; both are tool errors.
    with pytest.raises(ToolError):
        await retriever.search("anything", top_k=5)
