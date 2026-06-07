"""Agent picker — LLM chooses 1 of the probe's top-2 agents (Stage 4, §6.2).

The probe did the heavy lifting (5 -> 2); this small LLM call disambiguates.
Falls back to the probe's top-1 on any parse/LLM failure.
"""

import os

import weave
from litellm import completion

from src.crews.specialist import DEFAULT_BASE_URL, DEFAULT_MODEL


@weave.op(name="agent_picker")
def pick_agent(candidates: list[tuple[str, float]], query_text: str) -> str:
    """Pick one agent_id from the probe shortlist [(agent_id, score), ...]."""
    ids = [c[0] for c in candidates]
    if not ids:
        return ""
    if len(ids) == 1:
        return ids[0]

    prompt = (
        f"A SOC alert needs one specialist team.\n\nAlert: {query_text}\n\n"
        f"Choose the single best team from: {ids}\n"
        "Reply with only the exact team id, nothing else."
    )
    kwargs = {"model": DEFAULT_MODEL}
    if DEFAULT_MODEL.startswith(("ollama/", "ollama_chat/")):
        kwargs["api_base"] = DEFAULT_BASE_URL
    else:
        kwargs["api_key"] = os.environ.get("WANDB_API_KEY", "")

    try:
        response = completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
            **kwargs,
        )
        answer = (response.choices[0].message.content or "").strip()
        for cid in ids:
            if cid in answer:
                return cid
    except Exception:  # noqa: BLE001 — picker is best-effort; fall back to probe top-1
        pass
    return ids[0]
