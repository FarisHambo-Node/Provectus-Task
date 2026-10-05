"""Typed settings loaded from the environment.

Validation happens once at startup and raises `ConfigurationError`, so a bad
env var never surfaces as a mid-turn tool failure.
"""

from __future__ import annotations

import os
from typing import Final

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from agent.errors import ConfigurationError

_TRUE: Final = {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Everything the agent needs to boot, in one immutable object."""

    model_config = {"frozen": True}

    # --- AWS / Bedrock -----------------------------------------------------
    aws_region: str = "us-east-1"
    llm_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = Field(default=1024, ge=1)

    # --- OpenSearch --------------------------------------------------------
    opensearch_host: str = ""
    opensearch_index: str = "rag-chunks"
    # "es" for a managed domain, "aoss" for Serverless. Drives the SigV4 service name.
    opensearch_service: str = "es"

    # --- Retrieval ---------------------------------------------------------
    # top_k=5 keeps the synthesis prompt small and cheap; raising it improves
    # recall on multi-hop questions at the cost of tokens and dilution.
    top_k: int = Field(default=5, ge=1, le=50)
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    min_similarity: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Tool execution ----------------------------------------------------
    tool_timeout_s: float = Field(default=8.0, gt=0)
    tool_max_attempts: int = Field(default=3, ge=1, le=10)
    tool_base_backoff_s: float = Field(default=0.2, gt=0)
    tool_max_backoff_s: float = Field(default=2.0, gt=0)
    # boto3 is blocking, so every AWS call burns a thread: cap the fan-out.
    tool_max_concurrency: int = Field(default=4, ge=1, le=64)
    breaker_failure_threshold: int = Field(default=4, ge=1)
    breaker_reset_after_s: float = Field(default=20.0, gt=0)

    # --- Orchestration -----------------------------------------------------
    enable_web_search: bool = True
    max_tool_calls_per_turn: int = Field(default=4, ge=1)
    # Turns kept verbatim in the prompt before older ones get summarised.
    history_window_turns: int = Field(default=6, ge=1)

    # --- Simulation knobs (skeleton only, remove once AWS is wired in) -----
    simulate_latency: bool = True
    simulated_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    random_seed: int | None = None

    # --- Observability -----------------------------------------------------
    log_level: str = "INFO"
    log_format: str = "json"

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> "Settings":
        if load_dotenv_file:
            load_dotenv(override=False)
        raw = {
            "aws_region": os.getenv("AWS_REGION"),
            "llm_model_id": os.getenv("LLM_MODEL_ID"),
            "embedding_model_id": os.getenv("EMBEDDING_MODEL_ID"),
            "embedding_dimensions": os.getenv("EMBEDDING_DIMENSIONS"),
            "opensearch_host": os.getenv("OPENSEARCH_HOST"),
            "opensearch_index": os.getenv("OPENSEARCH_INDEX"),
            "opensearch_service": os.getenv("OPENSEARCH_SERVICE"),
            "top_k": os.getenv("TOP_K"),
            "chunk_size": os.getenv("CHUNK_SIZE"),
            "chunk_overlap": os.getenv("CHUNK_OVERLAP"),
            "min_similarity": os.getenv("MIN_SIMILARITY"),
            "tool_timeout_s": os.getenv("TOOL_TIMEOUT_S"),
            "tool_max_attempts": os.getenv("TOOL_MAX_ATTEMPTS"),
            "tool_max_concurrency": os.getenv("TOOL_MAX_CONCURRENCY"),
            "enable_web_search": _as_bool(os.getenv("ENABLE_WEB_SEARCH")),
            "max_tool_calls_per_turn": os.getenv("MAX_TOOL_CALLS_PER_TURN"),
            "history_window_turns": os.getenv("HISTORY_WINDOW_TURNS"),
            "simulate_latency": _as_bool(os.getenv("SIMULATE_LATENCY")),
            "simulated_failure_rate": os.getenv("SIMULATED_FAILURE_RATE"),
            "random_seed": os.getenv("RANDOM_SEED"),
            "log_level": os.getenv("LOG_LEVEL"),
            "log_format": os.getenv("LOG_FORMAT"),
        }
        present = {k: v for k, v in raw.items() if v is not None and v != ""}
        try:
            settings = cls(**present)
        except ValidationError as exc:
            raise ConfigurationError(
                "invalid environment configuration",
                context={"errors": exc.errors(include_url=False)},
            ) from exc
        if settings.chunk_overlap >= settings.chunk_size:
            raise ConfigurationError(
                "chunk_overlap must be smaller than chunk_size",
                context={
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                },
            )
        if settings.opensearch_service not in {"es", "aoss"}:
            raise ConfigurationError(
                "opensearch_service must be 'es' (domain) or 'aoss' (serverless)",
                context={"value": settings.opensearch_service},
            )
        return settings


def _as_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() in _TRUE
