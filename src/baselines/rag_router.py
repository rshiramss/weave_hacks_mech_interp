"""Arm 2 — RAG shortlist router (Stage 5 hybrid, §8).

Retrieve top-k tools by cosine similarity to the query (over Modal-precomputed
vectors), map the winning tool to its agent, and run that specialist crew
locally with the retrieved tools. `rag_retrieve` is a @weave.op so the retrieved
tool ids show in the trace.
"""

import weave

from src.baselines.embeddings import agent_of_tool, top_k_tools
from src.crews.specialist import MAX_TOOLS, build_specialist_crew
from src.probe.query_contract import normalize_query

RETRIEVE_K = 8  # retrieve a few extra, then filter to the winning agent


@weave.op(name="rag_retrieve")
def rag_retrieve(query_text: str, k: int = RETRIEVE_K) -> list[tuple[str, float]]:
    """Top-k tool_ids by cosine similarity. Logged to Weave."""
    return top_k_tools(query_text, k)


@weave.op(name="rag_router")
def rag_router(query_text: str, max_tools: int = MAX_TOOLS) -> dict:
    """Retrieve → pick agent from top tool → run that crew with retrieved tools."""
    query_text = normalize_query(query_text)
    retrieved = rag_retrieve(query_text)
    top_tool, _top_score = retrieved[0]

    agent_of = agent_of_tool()
    agent_id = agent_of[top_tool]

    tool_ids = [tid for tid, _ in retrieved if agent_of[tid] == agent_id][:max_tools]
    if top_tool not in tool_ids:
        tool_ids = [top_tool, *tool_ids][:max_tools]

    crew = build_specialist_crew(agent_id, tool_ids)
    output = crew.kickoff(inputs={"context": query_text})
    raw = getattr(output, "raw", None) or str(output)

    return {
        "arm": "rag",
        "agent_id": agent_id,
        "tool_id": top_tool,
        "tool_shortlist": tool_ids,
        "retrieved": retrieved,
        "result": raw,
    }
