"""Mock @tool stubs generated from data/registry.json (Stage 3, §6.5).

Every tool is a deterministic mock — no external APIs. Fair comparison across
all baseline arms (§6.5). Routing accuracy depends on labels, not execution
realism, so the bodies are intentionally stubs.

The crew factory only ever builds the probe-picked shortlist (≤5), so a
specialist never sees the full 200-tool catalog.

Reads `data/registry.json` directly (schema in §3 / §6.7) so Stage 3 stands
alone without the Stage 0 `src.registry` loader.
"""

import json
from functools import lru_cache
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# repo_root/src/tools/factory.py -> parents[2] == repo root
_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "registry.json"


@lru_cache(maxsize=1)
def _registry() -> dict:
    return json.loads(_REGISTRY_PATH.read_text())


@lru_cache(maxsize=1)
def _tools_by_id() -> dict:
    """tool_id -> registry entry {tool_id, agent_id, name, description, integration}."""
    return {entry["tool_id"]: entry for entry in _registry()["tools"]}


class MockToolInput(BaseModel):
    """Generic argument bag — a mock accepts whatever indicators the agent has."""

    indicators: str = Field(
        default="",
        description="Relevant indicators for this lookup: IPs, hostnames, file "
        "hashes, usernames, domains, message ids, etc.",
    )


class MockTool(BaseTool):
    """A registry-backed mock tool. Returns a deterministic JSON stub."""

    tool_id: str = ""
    agent_id: str = ""
    args_schema: type[BaseModel] = MockToolInput

    def _run(self, indicators: str = "", **kwargs) -> str:
        # Accept any extra kwargs the LLM hallucinates (it sometimes passes whole
        # JSON blobs) so a malformed call returns a mock result instead of erroring.
        extra = {k: v for k, v in kwargs.items() if v not in ("", None)}
        return json.dumps(
            {
                "tool_id": self.tool_id,
                "agent_id": self.agent_id,
                "status": "ok",
                "mock": True,
                "indicators": indicators or extra or None,
                "note": f"Mock result for {self.name}; no live integration.",
            }
        )


def build_tool(tool_id: str) -> MockTool:
    """Build one mock tool from its registry entry. Raises on unknown tool_id."""
    entry = _tools_by_id().get(tool_id)
    if entry is None:
        raise KeyError(f"Unknown tool_id '{tool_id}' (not in registry.json)")
    return MockTool(
        name=tool_id,  # callable identifier the LLM invokes
        description=f"{entry['name']}: {entry['description']}",
        tool_id=tool_id,
        agent_id=entry["agent_id"],
    )


def build_tools(tool_ids: list[str]) -> list[MockTool]:
    """Build mock tools for a shortlist of tool_ids, preserving order."""
    return [build_tool(tool_id) for tool_id in tool_ids]
