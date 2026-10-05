"""Tool layer: uniform contract, resilience, and the concrete capabilities."""

from agent.tools.base import Tool, ToolRequest, ToolResult, ToolSpec
from agent.tools.rag_tool import RAG_TOOL_SPEC, RagSearchTool
from agent.tools.registry import ToolRegistry
from agent.tools.resilience import CircuitBreaker, ResiliencePolicy, call_with_resilience
from agent.tools.web_search_tool import (
    WEB_SEARCH_TOOL_SPEC,
    SimulatedWebBackend,
    WebSearchBackend,
    WebSearchTool,
)

__all__ = [
    "RAG_TOOL_SPEC",
    "WEB_SEARCH_TOOL_SPEC",
    "CircuitBreaker",
    "RagSearchTool",
    "ResiliencePolicy",
    "SimulatedWebBackend",
    "Tool",
    "ToolRegistry",
    "ToolRequest",
    "ToolResult",
    "ToolSpec",
    "WebSearchBackend",
    "WebSearchTool",
    "call_with_resilience",
]
