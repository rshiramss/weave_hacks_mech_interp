"""Arm 1 — In-context router (Stage 5 hybrid, §8).

The standard production pattern: one LLM with the WHOLE 200-tool catalog in its
prompt, asked to pick the specialist agent + tools in one shot. No probes, no
activations, no masking. Selection + token counts are captured at this local
@weave.op adapter; the chosen agent + tools then run the same Stage 3 mock crew
locally so Weave auto-traces the crew subtree.

LLM backend: local Ollama Qwen2.5-7B-Instruct by default (W&B Inference does not
serve Qwen2.5-7B). Override INCONTEXT_MODEL / INCONTEXT_BASE_URL for a hosted
endpoint; an OpenAI-style model then reads OPENAI_API_KEY instead of api_base.
"""

import json
import os
import re
from functools import lru_cache

import weave
from litellm import completion

from src.crews.specialist import MAX_TOOLS, build_specialist_crew
from src.probe.query_contract import normalize_query
from src.tools.factory import _registry

INCONTEXT_MODEL = os.environ.get("INCONTEXT_MODEL", "ollama_chat/qwen2.5:7b-instruct")
INCONTEXT_BASE_URL = os.environ.get("INCONTEXT_BASE_URL", "http://localhost:11434")

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_SYSTEM_PROMPT = (
    "You are a SOC triage orchestrator. Given an analyst alert and the full tool "
    "catalog, choose the single best specialist agent and up to five tools to "
    'investigate. Respond with JSON only: {"agent_id": "<id>", "tool_ids": '
    '["<id>", ...]}.'
)


def _completion_kwargs() -> dict:
    """litellm kwargs: local ollama uses api_base + no key; hosted uses a key."""
    kwargs: dict = {"model": INCONTEXT_MODEL}
    if INCONTEXT_MODEL.startswith(("ollama/", "ollama_chat/")):
        kwargs["api_base"] = INCONTEXT_BASE_URL
    else:
        kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
    return kwargs


@lru_cache(maxsize=1)
def _catalog_text() -> str:
    """All 200 tools as `tool_id | agent=... | description` lines for the prompt."""
    return "\n".join(
        f"{t['tool_id']} | agent={t['agent_id']} | {t.get('description', '')}"
        for t in _registry()["tools"]
    )


@lru_cache(maxsize=1)
def _tools_for_agent() -> dict:
    mapping: dict[str, list[str]] = {}
    for tool in _registry()["tools"]:
        mapping.setdefault(tool["agent_id"], []).append(tool["tool_id"])
    return mapping


def _parse_and_validate(content: str, max_tools: int) -> tuple[str, list[str]]:
    """Parse model JSON and validate ids against the registry (never invent)."""
    agent_id, tool_ids = None, []
    match = _JSON_OBJECT.search(content or "")
    if match:
        try:
            data = json.loads(match.group(0))
            agent_id = data.get("agent_id")
            raw = data.get("tool_ids") or []
            if isinstance(raw, list):
                tool_ids = [str(t) for t in raw]
        except json.JSONDecodeError:
            pass

    valid_by_agent = _tools_for_agent()
    if agent_id not in valid_by_agent:  # recover agent from first valid tool
        for tid in tool_ids:
            for aid, owned in valid_by_agent.items():
                if tid in owned:
                    agent_id = aid
                    break
            if agent_id in valid_by_agent:
                break
    if agent_id not in valid_by_agent:
        raise ValueError(f"No valid agent in model output: {content!r}")

    owned = set(valid_by_agent[agent_id])
    chosen = [tid for tid in tool_ids if tid in owned][:max_tools]
    if not chosen:  # valid agent but unusable tools -> its real tools
        chosen = valid_by_agent[agent_id][:max_tools]
    return agent_id, chosen


@weave.op(name="incontext_route")
def incontext_route(query_text: str, max_tools: int = MAX_TOOLS) -> dict:
    """Full-catalog LLM selection. Returns {agent_id, tool_ids, token counts}."""
    response = completion(
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user", "content": f"Alert:\n{query_text}\n\n"
                   f"Tool catalog (tool_id | agent | description):\n{_catalog_text()}\n\n"
                   "Return JSON only."}],
        temperature=0,
        max_tokens=200,
        **_completion_kwargs(),
    )
    content = response.choices[0].message.content or ""
    agent_id, tool_ids = _parse_and_validate(content, max_tools)
    usage = response.usage
    return {
        "agent_id": agent_id,
        "tool_ids": tool_ids,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "model": INCONTEXT_MODEL,
    }


@weave.op(name="incontext_router")
def incontext_router(query_text: str) -> dict:
    """Arm 1 entry point: route in-context, then run the crew locally."""
    query_text = normalize_query(query_text)
    selection = incontext_route(query_text)
    agent_id, tool_ids = selection["agent_id"], selection["tool_ids"]

    crew = build_specialist_crew(agent_id, tool_ids)
    output = crew.kickoff(inputs={"context": query_text})
    raw = getattr(output, "raw", None) or str(output)

    return {
        "arm": "incontext",
        "agent_id": agent_id,
        "tool_id": tool_ids[0] if tool_ids else "",
        "tool_shortlist": tool_ids,
        "prompt_tokens": selection["prompt_tokens"],
        "completion_tokens": selection["completion_tokens"],
        "result": raw,
    }
