"""Routing-only adapters for the Weave Evaluation (Stage 6, §8).

Eval scores ROUTING (agent/tool correctness), not crew output, so these adapters
return the routing decision WITHOUT the slow local crew kickoff — keeping the
benchmark fast. The interactive `src/baselines/*_router` functions keep the full
routing+crew path for demos.

Output schema (all arms): {arm, agent_id, tool_id, tool_shortlist, latency_ms, ...}.
Results are memoized per (arm, query) so the Weave Evaluation and the wandb.Table
share one call per row.
"""

import time

import weave

from src.crews.specialist import MAX_TOOLS
from src.probe.query_contract import normalize_query

_CACHE: dict[tuple, dict] = {}


@weave.op(name="incontext_router")
def incontext_router(query_text: str) -> dict:
    """Arm 1 — full-catalog in-context selection (no crew)."""
    key = ("incontext", query_text)
    if key in _CACHE:
        return _CACHE[key]

    from src.baselines.frontier_incontext import incontext_route

    query = normalize_query(query_text)
    start = time.perf_counter()
    # Never raise inside an eval row: a failed LLM call would make this row's
    # output a Weave error-dict and break output summarization across rows.
    try:
        selection = incontext_route(query)
    except Exception:  # noqa: BLE001 — record a clean miss instead of an error row
        selection = {"agent_id": "", "tool_ids": [], "prompt_tokens": 0,
                     "completion_tokens": 0}
    tool_ids = selection.get("tool_ids") or []
    out = {
        "arm": "incontext",
        "agent_id": selection.get("agent_id") or "",
        "tool_id": tool_ids[0] if tool_ids else "",
        "tool_shortlist": tool_ids,
        "prompt_tokens": int(selection.get("prompt_tokens") or 0),
        "completion_tokens": int(selection.get("completion_tokens") or 0),
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "result": "",
    }
    _CACHE[key] = out
    return out


@weave.op(name="rag_router")
def rag_router(query_text: str) -> dict:
    """Arm 2 — cosine retrieval → agent (no crew)."""
    key = ("rag", query_text)
    if key in _CACHE:
        return _CACHE[key]

    from src.baselines.embeddings import agent_of_tool, top_k_tools

    query = normalize_query(query_text)
    start = time.perf_counter()
    retrieved = top_k_tools(query, 8)
    agent_of = agent_of_tool()
    top_tool = retrieved[0][0]
    agent_id = agent_of[top_tool]
    tool_ids = [tid for tid, _ in retrieved if agent_of[tid] == agent_id][:MAX_TOOLS]
    if top_tool not in tool_ids:
        tool_ids = [top_tool, *tool_ids][:MAX_TOOLS]

    out = {
        "arm": "rag",
        "agent_id": agent_id,
        "tool_id": top_tool,
        "tool_shortlist": tool_ids,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "result": "",
    }
    _CACHE[key] = out
    return out


@weave.op(name="probe_router")
def probe_router(query_text: str) -> dict:
    """Arm 3 — probe routing (no crew). Emits probe candidates for recall scorers."""
    key = ("probe", query_text)
    if key in _CACHE:
        return _CACHE[key]

    import time

    from src.probes.runtime import probe_route

    query = normalize_query(query_text)
    start = time.perf_counter()
    routed = probe_route(query)
    out = {
        "arm": "probe",
        "agent_id": routed["agent_id"],
        "tool_id": routed["tool_ids"][0] if routed["tool_ids"] else "",
        "tool_shortlist": routed["tool_ids"],
        "agent_candidates": routed["agent_candidates"],
        "tool_candidates": routed["tool_candidates"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "result": "",
    }
    _CACHE[key] = out
    return out


ROUTERS = {
    "incontext": incontext_router,
    "rag": rag_router,
    "probe": probe_router,
}

DISPLAY_NAMES = {
    "incontext": "Arm 1 — Full-catalog in-context",
    "rag": "Arm 2 — RAG",
    "probe": "Arm 3 — Probe router",
}
