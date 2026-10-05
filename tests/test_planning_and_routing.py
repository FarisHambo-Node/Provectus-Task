"""Tool selection and conditional-edge routing, with no AWS and no graph."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.nodes.analyze import keyword_score, looks_like_follow_up, plan_tools, resolve_query
from agent.routing import (
    NODE_ANALYZE,
    NODE_CLARIFY,
    NODE_FAILURE,
    NODE_SYNTHESIZE,
    route_after_analysis,
    route_after_gather,
    route_after_prepare,
)
from agent.settings import Settings
from agent.state import RetrievedChunk, ToolName, ToolPlan, TurnStatus
from agent.tools.rag_tool import RAG_TOOL_SPEC
from agent.tools.web_search_tool import WEB_SEARCH_TOOL_SPEC

ALL_SPECS = [RAG_TOOL_SPEC, WEB_SEARCH_TOOL_SPEC]


def plan(query: str, **overrides) -> ToolPlan:
    return plan_tools(query, specs=ALL_SPECS, settings=Settings(**overrides))


# --- planner ---------------------------------------------------------------


def test_internal_question_picks_rag_only():
    result = plan("where is our chunking configuration documented?")
    assert result.tools == [ToolName.RAG_SEARCH]


def test_current_events_question_includes_web():
    result = plan("what is the latest Bedrock pricing announced this week?")
    assert ToolName.WEB_SEARCH in result.tools


def test_mixed_question_selects_both_tools():
    result = plan("how do we chunk docs and what is the latest OpenSearch release?")
    assert set(result.tools) == {ToolName.RAG_SEARCH, ToolName.WEB_SEARCH}


def test_unmatched_question_falls_back_to_default_tool():
    result = plan("explain reciprocal degradation in sparse manifolds")
    assert result.tools == [ToolName.RAG_SEARCH]
    assert "default" in result.rationale


def test_short_query_requests_clarification():
    result = plan("hi")
    assert result.tools == []
    assert result.needs_clarification is True


def test_web_search_can_be_disabled():
    result = plan("latest pricing news today", enable_web_search=False)
    assert ToolName.WEB_SEARCH not in result.tools


def test_plan_respects_tool_call_budget():
    result = plan(
        "latest internal pricing docs and news", max_tool_calls_per_turn=1
    )
    assert len(result.tools) == 1


def test_keyword_score_is_bounded():
    assert 0.0 <= keyword_score("internal docs policy runbook", RAG_TOOL_SPEC) <= 1.0
    assert keyword_score("", WEB_SEARCH_TOOL_SPEC) == 0.0


def test_follow_up_detection():
    assert looks_like_follow_up("what about that one?") is True
    assert looks_like_follow_up("describe the ingestion pipeline") is False


def test_follow_up_query_is_made_self_contained():
    history = [HumanMessage(content="how do we embed documents?"), HumanMessage(content="what about that?")]
    resolved = resolve_query("what about that?", history)
    assert "how do we embed documents?" in resolved


# --- routing ---------------------------------------------------------------


def test_empty_query_skips_planning():
    assert route_after_prepare({"query": "   "}) == NODE_CLARIFY
    assert route_after_prepare({"query": "a real question"}) == NODE_ANALYZE


def test_routing_fans_out_to_every_planned_tool():
    state = {"plan": ToolPlan(tools=[ToolName.RAG_SEARCH, ToolName.WEB_SEARCH])}
    assert route_after_analysis(state) == ["rag_search", "web_search"]


def test_routing_without_a_plan_asks_for_clarification():
    assert route_after_analysis({}) == [NODE_CLARIFY]
    assert route_after_analysis({"plan": ToolPlan(tools=[])}) == [NODE_CLARIFY]


def test_clarification_status_overrides_tool_plan():
    state = {
        "plan": ToolPlan(tools=[ToolName.RAG_SEARCH]),
        "status": TurnStatus.NEEDS_CLARIFICATION,
    }
    assert route_after_analysis(state) == [NODE_CLARIFY]


def test_gather_routes_to_synthesis_with_evidence():
    state = {
        "status": TurnStatus.OK,
        "chunks": [
            RetrievedChunk(
                chunk_id="a#0",
                doc_id="a",
                title="t",
                source_uri="s3://a",
                text="body",
                score=0.7,
            )
        ],
    }
    assert route_after_gather(state) == NODE_SYNTHESIZE


def test_gather_routes_to_failure_without_evidence():
    assert route_after_gather({"status": TurnStatus.FAILED}) == NODE_FAILURE
    # Status says ok but nothing was retrieved: never let that reach synthesis.
    assert route_after_gather({"status": TurnStatus.OK}) == NODE_FAILURE
