"""Tool registry: the planner's view of what the agent can do."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from agent.errors import ToolNotFoundError
from agent.state import ToolName
from agent.tools.base import Tool, ToolSpec


class ToolRegistry:
    """Name-to-tool lookup plus the spec list handed to the planner."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[ToolName, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: ToolName | str) -> Tool:
        try:
            return self._tools[ToolName(name)]
        except (KeyError, ValueError) as exc:
            raise ToolNotFoundError(
                "no such tool",
                tool=str(name),
                context={"available": [str(n) for n in self._tools]},
            ) from exc

    def has(self, name: ToolName | str) -> bool:
        try:
            return ToolName(name) in self._tools
        except ValueError:
            return False

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def names(self) -> list[ToolName]:
        return list(self._tools)

    def to_bedrock_tool_config(self) -> dict[str, object]:
        """`toolConfig` payload for Bedrock Converse tool-choice planning."""
        return {"tools": [{"toolSpec": spec.to_bedrock_tool_spec()} for spec in self.specs()]}

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
