"""Routing scorers for the three-arm benchmark (Stage 6, §8).

Each scorer receives the gold dataset columns (agent_id, tool_id) plus the arm's
`output` dict. Probe-specific metrics (agent_recall_at_2, tool_recall_at_5)
return None when the output lacks probe metadata so non-probe arms simply skip
them instead of scoring 0 (Stage 6 note).
"""

import weave

# Rough W&B Inference / local pricing for the routing LLM; $/token.
_PROMPT_COST_PER_TOKEN = 0.20 / 1_000_000
_COMPLETION_COST_PER_TOKEN = 0.20 / 1_000_000


def estimate_cost(prompt_tokens, completion_tokens) -> float:
    p = prompt_tokens or 0
    c = completion_tokens or 0
    return round(p * _PROMPT_COST_PER_TOKEN + c * _COMPLETION_COST_PER_TOKEN, 8)


@weave.op()
def routing_exact_match(agent_id: str, tool_id: str, output: dict) -> bool | None:
    """Both the agent and the primary tool match the gold label."""
    if not output:
        return None
    return bool(output.get("agent_id") == agent_id and output.get("tool_id") == tool_id)


@weave.op()
def agent_recall_at_2(agent_id: str, output: dict) -> bool | None:
    """Gold agent in the probe top-2. None for arms without probe candidates."""
    candidates = (output or {}).get("agent_candidates")
    if not candidates:
        return None
    return bool(agent_id in candidates[:2])


@weave.op()
def tool_recall_at_5(tool_id: str, output: dict) -> bool | None:
    """Gold tool in the probe (masked) top-5. None for arms without candidates."""
    candidates = (output or {}).get("tool_candidates")
    if not candidates:
        return None
    return bool(tool_id in candidates[:5])


@weave.op()
def tool_in_shortlist(tool_id: str, output: dict) -> bool | None:
    """Gold tool in the arm's executed shortlist (<=5) — comparable across arms."""
    shortlist = (output or {}).get("tool_shortlist")
    if not shortlist:
        return None
    return bool(tool_id in shortlist)


@weave.op()
def estimated_cost(output: dict) -> float:
    o = output or {}
    return estimate_cost(o.get("prompt_tokens"), o.get("completion_tokens"))


@weave.op()
def latency_ms(output: dict):
    return (output or {}).get("latency_ms")


# Ordered list passed to weave.Evaluation.
SCORERS = [
    routing_exact_match,
    agent_recall_at_2,
    tool_recall_at_5,
    tool_in_shortlist,
    estimated_cost,
    latency_ms,
]
