"""Arm 1 in-context routing on Modal -> W&B Inference (Stage 5 hybrid).

A Modal container holds the full 200-tool catalog (registry.json baked into the
image), builds the routing prompt, and calls W&B Serverless Inference
(Qwen/Qwen2.5-7B-Instruct, OpenAI-compatible). It parses + validates the
selection against the registry and returns a dict. CrewAI crew EXECUTION stays
local (src/baselines/frontier_incontext.py) so Weave auto-traces it.

Secret: WANDB_API_KEY built from the local env at deploy — same pattern as
modal_app/generate_dataset.py.

Deploy once:
    modal deploy modal_app/incontext_route.py
Local adapter then calls modal.Function.from_name(APP_NAME, "route_incontext").
"""

import json
import os
from pathlib import Path

import modal

APP_NAME = "baseline-incontext"
WANDB_BASE_URL = "https://api.inference.wandb.ai/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # served by W&B Inference
MAX_TOOLS = 5

_REGISTRY_LOCAL = Path(__file__).resolve().parents[1] / "data" / "registry.json"
_REGISTRY_REMOTE = "/registry.json"

app = modal.App(APP_NAME)

# Bake registry.json into the image so the function can build the catalog itself.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("openai>=1.40")
    .add_local_file(str(_REGISTRY_LOCAL), _REGISTRY_REMOTE, copy=True)
)

# Build the Secret from the local environment at deploy time.
if modal.is_local():
    from dotenv import load_dotenv

    load_dotenv()
    _secret_values = {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")}
    if os.environ.get("WANDB_PROJECT"):
        _secret_values["WANDB_PROJECT"] = os.environ["WANDB_PROJECT"]
    wandb_secret = modal.Secret.from_dict(_secret_values)
else:
    wandb_secret = modal.Secret.from_dict({})

_SYSTEM_PROMPT = (
    "You are a SOC triage orchestrator. Given an analyst alert and the full tool "
    "catalog, choose the single best specialist agent and up to five tools to "
    'investigate. Respond with JSON only: {"agent_id": "<id>", "tool_ids": '
    '["<id>", ...]}.'
)


def _load_registry() -> dict:
    return json.loads(Path(_REGISTRY_REMOTE).read_text())


def _catalog_text(registry: dict) -> str:
    return "\n".join(
        f"{t['tool_id']} | agent={t['agent_id']} | {t.get('description', '')}"
        for t in registry["tools"]
    )


def _parse_and_validate(content: str, registry: dict) -> tuple[str, list[str]]:
    """Parse the model JSON and validate ids against the registry (never invent)."""
    import re

    tools_by_agent: dict[str, list[str]] = {}
    for tool in registry["tools"]:
        tools_by_agent.setdefault(tool["agent_id"], []).append(tool["tool_id"])

    agent_id, tool_ids = None, []
    match = re.search(r"\{.*\}", content or "", re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            agent_id = data.get("agent_id")
            raw_tools = data.get("tool_ids") or []
            if isinstance(raw_tools, list):
                tool_ids = [str(t) for t in raw_tools]
        except json.JSONDecodeError:
            pass

    if agent_id not in tools_by_agent:  # recover agent from first valid tool
        for tid in tool_ids:
            for aid, owned in tools_by_agent.items():
                if tid in owned:
                    agent_id = aid
                    break
            if agent_id in tools_by_agent:
                break
    if agent_id not in tools_by_agent:
        raise ValueError(f"No valid agent in model output: {content!r}")

    owned = set(tools_by_agent[agent_id])
    chosen = [tid for tid in tool_ids if tid in owned][:MAX_TOOLS]
    if not chosen:  # valid agent but unusable tools -> its real tools
        chosen = tools_by_agent[agent_id][:MAX_TOOLS]
    return agent_id, chosen


@app.function(image=image, secrets=[wandb_secret], timeout=600)
def route_incontext(query_text: str, model: str = DEFAULT_MODEL) -> dict:
    """Full-catalog in-context selection. Returns {agent_id, tool_id, tool_ids, tokens}."""
    from openai import OpenAI

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY missing in container — check Modal Secret.")

    registry = _load_registry()
    user = (
        f"Alert:\n{query_text}\n\n"
        f"Tool catalog (tool_id | agent | description):\n{_catalog_text(registry)}\n\n"
        "Return JSON only."
    )
    client = OpenAI(base_url=WANDB_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=200,
    )
    content = response.choices[0].message.content or ""
    agent_id, tool_ids = _parse_and_validate(content, registry)
    usage = response.usage

    return {
        "agent_id": agent_id,
        "tool_id": tool_ids[0] if tool_ids else "",
        "tool_ids": tool_ids,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "model": model,
    }
