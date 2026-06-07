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


@weave.op(name="tool_probe")
def tool_probe(h, agent_id: str, k: int = 5) -> list[tuple[str, float]]:
    _agent, tool_probe_clf = load_probes()
    return tool_probe_clf.top_k_masked(h, agent_id, k)


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
