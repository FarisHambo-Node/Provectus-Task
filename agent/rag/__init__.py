"""Simulated internal document store, independent of tools and orchestration."""

from agent.rag.corpus import SAMPLE_CORPUS, CorpusDocument, corpus_size
from agent.rag.retriever import Retriever, SimulatedVectorRetriever

__all__ = [
    "SAMPLE_CORPUS",
    "CorpusDocument",
    "Retriever",
    "SimulatedVectorRetriever",
    "corpus_size",
]
