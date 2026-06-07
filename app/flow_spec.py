"""Canonical step graphs for the three router arms (front-end source of truth).

The UI renders this skeleton immediately, then lights each node up as live SSE
step events arrive (app/run_stream.py). Node ids here MUST match the `step_id`
values emitted by the runners so the front end can map events to nodes.

`execute_only` nodes (the crew) are only reached when a run requests crew
execution; the front end prunes them (and dangling edges) when execute=False.

The `route_agent` node (kind "agent") is a *display* node — no runner emits a
`step_id` for it. The front end fills it from the chosen agent carried in the
routing step's detail, showing the full roster of specialist sub-agents with the
routed one highlighted.
"""

# kind drives node styling on the front end: ingest | route | probe | agent | crew
_ROUTE_AGENT = {"id": "route_agent", "label": "Routed specialist", "kind": "agent"}

ARM_FLOWS: dict[str, dict] = {
    "incontext": {
        "label": "In-context (full catalog)",
        "nodes": [
            {"id": "normalize", "label": "Normalize query", "kind": "ingest"},
            {"id": "incontext_route",
             "label": "LLM select · 200-tool catalog", "kind": "route"},
            _ROUTE_AGENT,
            {"id": "build_crew", "label": "Build specialist crew",
             "kind": "crew", "execute_only": True},
            {"id": "crew_kickoff", "label": "Crew investigate",
             "kind": "crew", "execute_only": True},
        ],
        "edges": [
            ["normalize", "incontext_route"],
            ["incontext_route", "route_agent"],
            ["route_agent", "build_crew"],
            ["build_crew", "crew_kickoff"],
        ],
    },
    "rag": {
        "label": "RAG (cosine retrieval)",
        "nodes": [
            {"id": "normalize", "label": "Normalize query", "kind": "ingest"},
            {"id": "rag_retrieve", "label": "Embed + cosine top-k", "kind": "route"},
            {"id": "filter_to_agent", "label": "Filter to winning agent",
             "kind": "route"},
            _ROUTE_AGENT,
            {"id": "build_crew", "label": "Build specialist crew",
             "kind": "crew", "execute_only": True},
            {"id": "crew_kickoff", "label": "Crew investigate",
             "kind": "crew", "execute_only": True},
        ],
        "edges": [
            ["normalize", "rag_retrieve"],
            ["rag_retrieve", "filter_to_agent"],
            ["filter_to_agent", "route_agent"],
            ["route_agent", "build_crew"],
            ["build_crew", "crew_kickoff"],
        ],
    },
    "probe": {
        "label": "Probe (layer-24 linear probe)",
        "nodes": [
            {"id": "normalize", "label": "Normalize query", "kind": "ingest"},
            {"id": "forward_pass", "label": "Forward pass · Modal GPU h",
             "kind": "probe"},
            {"id": "agent_probe", "label": "Agent probe · top-2", "kind": "probe"},
            {"id": "agent_picker", "label": "LLM picks 1 of 2", "kind": "probe"},
            {"id": "tool_probe", "label": "Tool probe · per-agent top-5",
             "kind": "probe"},
            _ROUTE_AGENT,
            {"id": "build_crew", "label": "Build specialist crew",
             "kind": "crew", "execute_only": True},
            {"id": "crew_kickoff", "label": "Crew investigate",
             "kind": "crew", "execute_only": True},
        ],
        "edges": [
            ["normalize", "forward_pass"],
            ["forward_pass", "agent_probe"],
            ["agent_probe", "agent_picker"],
            ["agent_picker", "tool_probe"],
            ["tool_probe", "route_agent"],
            ["route_agent", "build_crew"],
            ["build_crew", "crew_kickoff"],
        ],
    },
}

ARMS = tuple(ARM_FLOWS.keys())


def _agent_roster() -> dict[str, str]:
    """agent_id -> specialty (single source of truth: the crew factory).

    Imported lazily so callers that only need ARMS don't pull in CrewAI.
    """
    from src.crews.specialist import SPECIALTY

    return dict(SPECIALTY)


def flow_spec() -> dict:
    """The full per-arm node/edge spec served at GET /flow-spec.

    Also carries the specialist `agents` roster (for the routed-agent node and
    legend) and the `weave_project` (for the per-arm Weave trace link).
    """
    from src.weave_setup import WEAVE_PROJECT

    return {
        "arms": list(ARMS),
        "flows": ARM_FLOWS,
        "agents": _agent_roster(),
        "weave_project": WEAVE_PROJECT,
    }
