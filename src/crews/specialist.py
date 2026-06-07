"""Specialist crew factory (Stage 3, agent_architecture.md §6.5).

One Crew per sub-agent, same template — only the specialty label and the
probe-picked tools differ. Routing is already done; the agent investigates.

v1 constraints (§6.5): allow_delegation=False, tools = probe top-5 only,
input = {context} only, one task, Process.sequential, max_iter=5.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from crewai import LLM, Agent, Crew, Process, Task

from src.tools.factory import build_tools

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# Probe tool shortlist size (§6.4 / §6.5). A crew never receives more than this.
MAX_TOOLS = 5

# Execution LLM. Default targets a LOCAL Ollama Qwen2.5-7B-Instruct (W&B
# Inference does not serve Qwen2.5-7B). Override via env for a hosted endpoint.
DEFAULT_MODEL = os.environ.get(
    "LLM_MODEL", "ollama_chat/qwen2.5:7b-instruct"
)
DEFAULT_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "http://localhost:11434"
)

# agent_id -> {specialty} string (§6.5 table).
SPECIALTY = {
    "log_search": "SIEM log search and correlation",
    "threat_intel": "threat intelligence and IOC enrichment",
    "malware_analysis": "malware and endpoint forensics",
    "network_analysis": "network traffic and DNS analysis",
    "email_security": "email security and phishing investigation",
}


@lru_cache(maxsize=2)
def _load_config(filename: str) -> dict:
    return yaml.safe_load((_CONFIG_DIR / filename).read_text())


def load_agents_config() -> dict:
    return _load_config("agents.yaml")


def load_tasks_config() -> dict:
    return _load_config("tasks.yaml")


def _default_llm() -> LLM:
    """LLM pointed at the configured endpoint.

    Construction makes no network call, so a crew can be *built* without
    credentials; the key is only needed at kickoff. An empty key is left as-is
    so build-only checks (acceptance #1) pass offline.
    """
    return LLM(
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key=os.environ.get("WANDB_API_KEY", ""),
        temperature=0,
    )


def build_specialist_crew(
    agent_id: str,
    tool_ids: list[str],
    agents_config: dict | None = None,
    tasks_config: dict | None = None,
    llm: LLM | None = None,
) -> Crew:
    """Build a single-specialist crew for `agent_id` with the given tool shortlist.

    `tool_ids` is the probe top-5 (≤5 enforced). Raises on unknown agent_id or
    an over-long shortlist. `{context}` stays unformatted — it is supplied at
    `crew.kickoff(inputs={"context": query_text})`.
    """
    if agent_id not in SPECIALTY:
        raise KeyError(
            f"Unknown agent_id '{agent_id}'. Known: {sorted(SPECIALTY)}"
        )
    if len(tool_ids) > MAX_TOOLS:
        raise ValueError(
            f"{len(tool_ids)} tools exceeds the probe shortlist cap of {MAX_TOOLS}"
        )

    agents_config = agents_config or load_agents_config()
    tasks_config = tasks_config or load_tasks_config()
    specialty = SPECIALTY[agent_id]

    template = agents_config["soc_specialist"]
    agent = Agent(
        role=template["role"].format(specialty=specialty),
        goal=template["goal"].format(specialty=specialty),
        backstory=template["backstory"].format(specialty=specialty),
        allow_delegation=template.get("allow_delegation", False),
        verbose=template.get("verbose", True),
        max_iter=template.get("max_iter", 5),
        tools=build_tools(list(tool_ids)),
        llm=llm or _default_llm(),
    )

    task_template = tasks_config["investigate_alert"]
    task = Task(
        description=task_template["description"],
        expected_output=task_template["expected_output"],
        agent=agent,
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
