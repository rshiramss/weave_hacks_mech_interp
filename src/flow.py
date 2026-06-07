"""ProbeRoutedFlow — single-hop probe router (Stage 4, §6.6).

CrewAI Flow: ingest -> (forward_pass -> agent_probe -> agent_picker ->
tool_probe) -> specialist crew kickoff. weave.init() auto-traces the Flow + Crew
subtree; the probe sub-steps are @weave.op (src/probes/runtime.py) so the full
tree is ingest_event -> forward_pass -> agent_probe -> agent_picker -> tool_probe
-> crewai.Crew.kickoff.
"""

import weave
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

from src.crews.specialist import build_specialist_crew
from src.probe.query_contract import normalize_query
from src.probes.runtime import agent_probe, forward_pass, tool_probe
from src.routing.agent_picker import pick_agent


class RouterState(BaseModel):
    raw_input: str = ""
    query_text: str = ""
    agent_candidates: list = []
    agent_id: str = ""
    tool_shortlist: list = []
    tool_id: str = ""
    result: str = ""


class ProbeRoutedFlow(Flow[RouterState]):
    @start()
    @weave.op(name="ingest_event")
    def ingest(self):
        self.state.query_text = normalize_query(self.state.raw_input)
        return self.state.query_text

    @listen(ingest)
    @weave.op(name="route_and_execute")
    def route_and_execute(self, query_text):
        h = forward_pass(query_text)
        candidates = agent_probe(h, 2)
        self.state.agent_candidates = [a for a, _ in candidates]
        self.state.agent_id = pick_agent(candidates, query_text)

        tools = tool_probe(h, self.state.agent_id, 5)
        self.state.tool_shortlist = [t for t, _ in tools]
        self.state.tool_id = self.state.tool_shortlist[0] if self.state.tool_shortlist else ""

        crew = build_specialist_crew(self.state.agent_id, self.state.tool_shortlist)
        output = crew.kickoff(inputs={"context": query_text})
        self.state.result = getattr(output, "raw", None) or str(output)
        return self.state


def run_probe_flow(query: str) -> dict:
    """Convenience wrapper: run the flow and return a plain dict."""
    flow = ProbeRoutedFlow()
    flow.kickoff(inputs={"raw_input": query})
    s = flow.state
    return {
        "arm": "probe",
        "query_text": s.query_text,
        "agent_id": s.agent_id,
        "tool_id": s.tool_id,
        "tool_shortlist": s.tool_shortlist,
        "agent_candidates": s.agent_candidates,
        "result": s.result,
    }
