"""Sample internal corpus for the simulated RAG system.

Twenty short documents standing in for the real OpenSearch index. The texts are
deliberately about the system itself so answers during a demo read sensibly.
"""

from __future__ import annotations

from pydantic import BaseModel


class CorpusDocument(BaseModel):
    """A pre-chunked document. One document == one chunk in this skeleton."""

    doc_id: str
    title: str
    section: str
    source_uri: str
    text: str
    # Cheap stand-in for a real embedding; the simulated retriever scores
    # lexical overlap against these instead of calling Titan.
    keywords: tuple[str, ...] = ()


SAMPLE_CORPUS: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        doc_id="doc-001",
        title="Ingestion pipeline overview",
        section="Architecture",
        source_uri="s3://internal-docs/platform/ingestion.md#overview",
        text=(
            "Documents land in the raw S3 bucket, a Lambda extracts text, and a "
            "Step Functions map state fans out chunking and embedding. Failed "
            "objects go to a DLQ and are replayed by the nightly backfill job."
        ),
        keywords=("ingestion", "pipeline", "s3", "lambda", "step functions", "dlq"),
    ),
    CorpusDocument(
        doc_id="doc-002",
        title="Chunking strategy and defaults",
        section="Retrieval",
        source_uri="s3://internal-docs/platform/chunking.md",
        text=(
            "Default chunk size is 1000 characters with 150 characters of overlap, "
            "split on paragraph boundaries first and sentences second. Larger "
            "chunks preserve context but dilute the embedding; smaller chunks "
            "retrieve precisely but lose cross-paragraph reasoning."
        ),
        keywords=("chunking", "chunk size", "overlap", "splitting", "defaults"),
    ),
    CorpusDocument(
        doc_id="doc-003",
        title="Titan Text Embeddings v2 configuration",
        section="Embeddings",
        source_uri="s3://internal-docs/platform/embeddings.md",
        text=(
            "We call amazon.titan-embed-text-v2:0 with normalize set to true and "
            "1024 dimensions. Normalisation lets us use cosine similarity in "
            "OpenSearch without rescaling at query time."
        ),
        keywords=("titan", "embedding", "normalize", "dimensions", "cosine"),
    ),
    CorpusDocument(
        doc_id="doc-004",
        title="OpenSearch k-NN index mapping",
        section="Vector store",
        source_uri="s3://internal-docs/platform/opensearch-mapping.md",
        text=(
            "The rag-chunks index uses a knn_vector field of dimension 1024 with "
            "the HNSW method, cosine similarity, m=16 and ef_construction=256. "
            "Metadata fields are keyword-typed so filters stay exact."
        ),
        keywords=("opensearch", "knn", "hnsw", "mapping", "index", "cosine"),
    ),
    CorpusDocument(
        doc_id="doc-005",
        title="Hybrid search and rank fusion",
        section="Retrieval",
        source_uri="s3://internal-docs/platform/hybrid-search.md",
        text=(
            "Dense k-NN and BM25 run as two queries and are merged with "
            "reciprocal rank fusion at k=60. Hybrid search wins on acronyms and "
            "product names that embeddings blur, at the cost of a second query "
            "per request."
        ),
        keywords=("hybrid", "bm25", "rank fusion", "rrf", "reranking"),
    ),
    CorpusDocument(
        doc_id="doc-006",
        title="SigV4 signing for OpenSearch",
        section="Security",
        source_uri="s3://internal-docs/platform/sigv4.md",
        text=(
            "Requests are signed with SigV4. The service name is 'es' for a "
            "managed domain and 'aoss' for OpenSearch Serverless; using the wrong "
            "one returns a 403 that looks like an auth misconfiguration."
        ),
        keywords=("sigv4", "signing", "aoss", "es", "auth", "403"),
    ),
    CorpusDocument(
        doc_id="doc-007",
        title="Bedrock Converse API usage",
        section="Generation",
        source_uri="s3://internal-docs/platform/converse.md",
        text=(
            "Answer generation uses the Bedrock Converse API with a system prompt "
            "that forbids answering outside the retrieved context. Tool use is "
            "declared through the toolConfig block."
        ),
        keywords=("bedrock", "converse", "generation", "system prompt", "tool use"),
    ),
    CorpusDocument(
        doc_id="doc-008",
        title="Throttling and retry policy",
        section="Reliability",
        source_uri="s3://internal-docs/platform/retries.md",
        text=(
            "Bedrock ThrottlingException is retried up to three times with "
            "exponential backoff and full jitter. A circuit breaker opens after "
            "four consecutive failures and half-opens twenty seconds later."
        ),
        keywords=("throttling", "retry", "backoff", "jitter", "circuit breaker"),
    ),
    CorpusDocument(
        doc_id="doc-009",
        title="Citation format contract",
        section="Generation",
        source_uri="s3://internal-docs/platform/citations.md",
        text=(
            "Every claim carries a bracketed marker such as [1] that maps to a "
            "citation entry with title and source URI. Answers without a marker "
            "are rejected by the post-check and regenerated once."
        ),
        keywords=("citation", "marker", "grounding", "post-check"),
    ),
    CorpusDocument(
        doc_id="doc-010",
        title="Evaluation harness",
        section="Quality",
        source_uri="s3://internal-docs/platform/eval.md",
        text=(
            "A golden set of 200 question/answer pairs runs nightly. We track "
            "recall@5 for retrieval and a faithfulness score for generation, and "
            "block deploys when either drops more than two points."
        ),
        keywords=("evaluation", "recall", "faithfulness", "golden set", "metrics"),
    ),
    CorpusDocument(
        doc_id="doc-011",
        title="Lambda versus ECS for the API",
        section="Deployment",
        source_uri="s3://internal-docs/platform/compute.md",
        text=(
            "Lambda is the default because traffic is bursty and idle cost is "
            "zero; ECS Fargate takes over when we need streaming responses and "
            "warm connection pools to OpenSearch."
        ),
        keywords=("lambda", "ecs", "fargate", "deployment", "compute", "streaming"),
    ),
    CorpusDocument(
        doc_id="doc-012",
        title="Multi-tenant access filtering",
        section="Security",
        source_uri="s3://internal-docs/platform/tenancy.md",
        text=(
            "Every chunk stores tenant_id and an ACL list. Retrieval always "
            "applies a pre-filter on the caller's tenant; post-filtering was "
            "dropped because it silently shrank the result set below top_k."
        ),
        keywords=("tenant", "acl", "filter", "security", "isolation"),
    ),
    CorpusDocument(
        doc_id="doc-013",
        title="Conversation memory and summarisation",
        section="Agent",
        source_uri="s3://internal-docs/platform/memory.md",
        text=(
            "The six most recent turns stay verbatim in the prompt. Older turns "
            "are folded into a rolling summary so a long thread does not push the "
            "retrieved context out of the window."
        ),
        keywords=("memory", "conversation", "summary", "history", "multi-turn"),
    ),
    CorpusDocument(
        doc_id="doc-014",
        title="Query rewriting for follow-ups",
        section="Agent",
        source_uri="s3://internal-docs/platform/query-rewrite.md",
        text=(
            "Pronouns and ellipsis in follow-up questions are resolved against "
            "the last two turns before retrieval. Unresolved references are the "
            "largest single source of empty result sets."
        ),
        keywords=("query rewriting", "follow-up", "pronoun", "coreference"),
    ),
    CorpusDocument(
        doc_id="doc-015",
        title="Tool routing policy",
        section="Agent",
        source_uri="s3://internal-docs/platform/routing.md",
        text=(
            "Internal retrieval runs for any product or policy question. Web "
            "search is added only when the query asks about current events, "
            "pricing, or third-party releases, because it costs an external call."
        ),
        keywords=("routing", "tool selection", "planner", "web search"),
    ),
    CorpusDocument(
        doc_id="doc-016",
        title="Cost model per thousand queries",
        section="Operations",
        source_uri="s3://internal-docs/platform/cost.md",
        text=(
            "At top_k=5 a thousand queries cost roughly four dollars, dominated "
            "by generation tokens. Doubling top_k adds about sixty percent to the "
            "prompt cost for a two point recall gain."
        ),
        keywords=("cost", "pricing", "tokens", "budget", "top_k"),
    ),
    CorpusDocument(
        doc_id="doc-017",
        title="Index refresh and reindexing",
        section="Operations",
        source_uri="s3://internal-docs/platform/reindex.md",
        text=(
            "Embedding model changes require a full reindex into a new alias "
            "target, then an atomic alias swap. Refresh interval is set to thirty "
            "seconds during bulk loads and back to one second afterwards."
        ),
        keywords=("reindex", "alias", "refresh", "bulk", "migration"),
    ),
    CorpusDocument(
        doc_id="doc-018",
        title="Observability and tracing",
        section="Operations",
        source_uri="s3://internal-docs/platform/observability.md",
        text=(
            "Each turn emits one structured log per node with thread_id and "
            "turn_index, plus CloudWatch EMF metrics for tool latency and failure "
            "counts. Traces are sampled at ten percent in production."
        ),
        keywords=("observability", "logging", "metrics", "tracing", "cloudwatch"),
    ),
    CorpusDocument(
        doc_id="doc-019",
        title="PII redaction before indexing",
        section="Security",
        source_uri="s3://internal-docs/platform/pii.md",
        text=(
            "Comprehend detects PII during ingestion and the pipeline replaces "
            "spans with typed placeholders before embedding, so the vector store "
            "never holds raw identifiers."
        ),
        keywords=("pii", "redaction", "comprehend", "privacy", "compliance"),
    ),
    CorpusDocument(
        doc_id="doc-020",
        title="Fallback behaviour when retrieval is empty",
        section="Reliability",
        source_uri="s3://internal-docs/platform/fallback.md",
        text=(
            "If no chunk clears the similarity floor the agent says it cannot "
            "answer and offers to widen the search. Guessing from parametric "
            "knowledge is treated as an incident."
        ),
        keywords=("fallback", "empty results", "refusal", "similarity floor"),
    ),
)


def corpus_size() -> int:
    return len(SAMPLE_CORPUS)
