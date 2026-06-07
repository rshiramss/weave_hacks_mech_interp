"""Weave-traced probe runtime (Stage 4, §6 / §7.1).

Thin @weave.op wrappers around the GPU forward pass (Modal extractor) and the two
probes so the routing steps appear in the trace tree. `probe_route` is the
routing-only entry used by the eval adapter and the demo dispatcher; the full
ProbeRoutedFlow (src/flow.py) calls the individual ops then runs the crew.
"""

import weave

from src.probes.extract_h import extract_h_remote
from src.probes.inference import load_probes
from src.routing.agent_picker import pick_agent


@weave.op(name="forward_pass")
def forward_pass(query_text: str):
    """Residual-stream vector h for the query (runs on the Modal GPU extractor)."""
    return extract_h_remote([query_text])[0]


@weave.op(name="agent_probe")
def agent_probe(h, k: int = 2) -> list[tuple[str, float]]:
    agent_probe_clf, _tool = load_probes()
    return agent_probe_clf.top_k(h, k)


def _tool_probe_display_name(call) -> str:
    """Trace each per-agent tool probe by the agent it ran for (e.g. tool_probe:log_search)."""
    try:
        return f"tool_probe:{(call.inputs or {}).get('agent_id', '?')}"
    except Exception:  # noqa: BLE001 — display name is cosmetic, never break the op
        return "tool_probe"


@weave.op(name="tool_probe", call_display_name=_tool_probe_display_name)
def tool_probe(h, agent_id: str, k: int = 5) -> list[tuple[str, float]]:
    """Top-5 tools from the chosen agent's own 40-class probe (no masking)."""
    _agent, tool_probes = load_probes()
    return tool_probes.top_k(h, agent_id, k)


@weave.op(name="probe_route")
def probe_route(query_text: str) -> dict:
    """One forward pass -> agent top-2 -> pick -> masked tool top-5 (no crew)."""
    h = forward_pass(query_text)
    candidates = agent_probe(h, 2)
    agent_id = pick_agent(candidates, query_text)
    tools = tool_probe(h, agent_id, 5)
    return {
        "agent_id": agent_id,
        "tool_ids": [t for t, _ in tools],
        "agent_candidates": [a for a, _ in candidates],
        "tool_candidates": [t for t, _ in tools],
    }
