"""Demo routing pipeline (Stage 8): structured alert -> NL -> route -> triage.

ProbeRoutedFlow (Stage 4) is not built yet, so this function-based dispatcher
provides the same end-to-end demo path now. Backend is chosen by `ROUTER_BACKEND`
(or the `backend` arg):

    stub       — instant keyword router over the registry (no LLM, offline)
    rag        — Arm 2 cosine retrieval (Modal embeddings)
    incontext  — Arm 1 full-catalog LLM (local Qwen)
    probe      — Arm 3 (raises until Stage 4 lands)

Swap to the real ProbeRoutedFlow / probe backend later; the return shape is
stable so the UI does not change.
"""

import os
import re
from functools import lru_cache

import weave

from src.crews.specialist import MAX_TOOLS, build_specialist_crew
from src.formatters import to_analyst_message
from src.tools.factory import _registry

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD.findall(text.lower()))


@lru_cache(maxsize=1)
def _tool_tokens() -> list[tuple[str, str, set]]:
    out = []
    for tool in _registry()["tools"]:
        text = (
            f"{tool['tool_id'].replace('_', ' ')} "
            f"{tool.get('name', '')} {tool.get('description', '')}"
        )
        out.append((tool["tool_id"], tool["agent_id"], _tokens(text)))
    return out


@weave.op(name="stub_route")
def _stub_route(query_text: str, max_tools: int = MAX_TOOLS) -> tuple[str, list[str]]:
    """Instant keyword-overlap router over the registry (no LLM)."""
    query = _tokens(query_text)
    scored = sorted(
        ((tid, aid, len(query & toks)) for tid, aid, toks in _tool_tokens()),
        key=lambda x: x[2],
        reverse=True,
    )
    top_tool, agent_id, _score = scored[0]
    tool_ids = [tid for tid, aid, _ in scored if aid == agent_id][:max_tools]
    if top_tool not in tool_ids:
        tool_ids = [top_tool, *tool_ids][:max_tools]
    return agent_id, tool_ids


def _route(backend: str, query_text: str) -> tuple[str, list[str]]:
    if backend == "stub":
        return _stub_route(query_text)
    if backend == "rag":
        from src.baselines.embeddings import agent_of_tool, top_k_tools

        retrieved = top_k_tools(query_text, 8)
        agent_of = agent_of_tool()
        top = retrieved[0][0]
        agent_id = agent_of[top]
        tool_ids = [tid for tid, _ in retrieved if agent_of[tid] == agent_id][:MAX_TOOLS]
        if top not in tool_ids:
            tool_ids = [top, *tool_ids][:MAX_TOOLS]
        return agent_id, tool_ids
    if backend == "incontext":
        from src.baselines.frontier_incontext import incontext_route

        selection = incontext_route(query_text)
        return selection["agent_id"], selection["tool_ids"]
    if backend == "probe":
        from src.probes.runtime import probe_route

        routed = probe_route(query_text)
        return routed["agent_id"], routed["tool_ids"]
    raise ValueError(f"Unknown ROUTER_BACKEND: {backend}")


@weave.op(name="triage_alert")
def route_alert(event, backend: str | None = None, execute: bool = True) -> dict:
    """Ingest a structured/NL alert, route it, and (optionally) run the crew."""
    backend = backend or os.environ.get("ROUTER_BACKEND", "stub")
    query_text = to_analyst_message(event)

    agent_id, tool_ids = _route(backend, query_text)

    result = ""
    if execute:
        crew = build_specialist_crew(agent_id, tool_ids)
        output = crew.kickoff(inputs={"context": query_text})
        result = getattr(output, "raw", None) or str(output)

    return {
        "backend": backend,
        "query_text": query_text,
        "agent_id": agent_id,
        "tool_id": tool_ids[0] if tool_ids else "",
        "tool_shortlist": tool_ids,
        "result": result,
    }
