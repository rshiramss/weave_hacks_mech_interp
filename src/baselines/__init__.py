"""Baseline routers (Stage 5, agent_architecture.md §8).

Shared output contract for ALL arms (1 frontier, 2 RAG, 3 probe) so the Stage 6
scorers can treat them identically:

    {
        "arm": "incontext" | "rag" | "probe",
        "agent_id": str,            # chosen specialist
        "tool_id": str,             # primary routed tool (first of shortlist)
        "tool_shortlist": [str],    # tools handed to the crew (<=5)
        "result": str,              # crew JSON triage output (raw)
        ...                         # arm-specific extras (retrieval scores, etc.)
    }

Every arm executes the SAME mock-tool Stage 3 crews, so the comparison axis is
routing method + cost, not execution.
"""

OUTPUT_KEYS = ("arm", "agent_id", "tool_id", "tool_shortlist", "result")
