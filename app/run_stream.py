"""Live run streaming for the front end (Phase 1).

Runs the selected router arms CONCURRENTLY (one worker thread each, since the
underlying steps are synchronous + blocking — Modal, Ollama, sklearn) and streams
a step event as every stage starts/finishes. The front end maps each event's
`step_id` onto the canonical graph nodes in app/flow_spec.py and lights them up.

Event kinds (all JSON on the default SSE channel):
  run_started  {run_id, arms, execute}
  step         {run_id, arm, step_id, status: started|done|error, t_ms, detail}
  arm_done     {run_id, arm, ok, latency_ms, cost_usd, agent_id, tool_id,
                tool_shortlist, result}
  run_done     {run_id}

The runners only call existing functions (src/probes/runtime, src/baselines/*,
src/crews/specialist) — no routing logic is re-implemented here.
"""

import asyncio
import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from app.flow_spec import ARMS
from eval.scorers import estimate_cost
from src.probe.query_contract import normalize_query

# Cap the crew result echoed to the UI so a chatty agent can't bloat the stream.
MAX_RESULT_CHARS = 2000


def available_arms() -> tuple[str, ...]:
    return ARMS


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _pairs(pairs: list) -> list[dict]:
    """[(id, score), ...] -> [{"id", "score"}] for JSON display."""
    return [{"id": str(i), "score": round(float(s), 4)} for i, s in pairs]


# --- per-step emit + timing -------------------------------------------------


@contextmanager
def _step(emit_step: Callable, step_id: str) -> Iterator[dict]:
    """Emit started -> (fill detail) -> done; on error emit error and re-raise."""
    t0 = time.perf_counter()
    emit_step(step_id, "started", None, None)
    detail: dict = {}
    try:
        yield detail
    except Exception as exc:  # noqa: BLE001 — surface failure as an error event
        emit_step(step_id, "error", {"error": str(exc)}, _ms(t0))
        raise
    emit_step(step_id, "done", detail or None, _ms(t0))


def _maybe_crew(execute: bool, step, agent_id: str, tool_ids: list[str],
                query_text: str) -> str:
    if not execute:
        return ""
    from src.crews.specialist import build_specialist_crew

    with step("build_crew"):
        crew = build_specialist_crew(agent_id, tool_ids)
    raw = ""
    with step("crew_kickoff") as detail:
        output = crew.kickoff(inputs={"context": query_text})
        raw = getattr(output, "raw", None) or str(output)
        detail["result"] = raw[:MAX_RESULT_CHARS]
    return raw


# --- arm runners: (query, execute, step) -> (agent_id, tool_ids, cost, result)


def _run_probe(query: str, execute: bool, step):
    from src.probes.inference import load_probes
    from src.probes.runtime import agent_probe, forward_pass, tool_probe
    from src.routing.agent_picker import pick_agent

    with step("normalize") as detail:
        query_text = normalize_query(query)
        detail["query_text"] = query_text
    with step("forward_pass") as detail:
        h = forward_pass(query_text)
        detail["dim"] = int(len(h))
    with step("agent_probe") as detail:
        candidates = agent_probe(h, 2)
        detail["candidates"] = _pairs(candidates)
    with step("agent_picker") as detail:
        agent_id = pick_agent(candidates, query_text)
        detail["chosen"] = agent_id
    with step("tool_probe") as detail:
        tools = tool_probe(h, agent_id, 5)
        detail["tools"] = _pairs(tools)
        detail["probe_agent"] = agent_id  # which of the per-agent probes fired
        _, tool_probes = load_probes()  # lru_cached; cheap re-lookup
        detail["pool_size"] = tool_probes.pool_size(agent_id)  # scoped tool count
    tool_ids = [t for t, _ in tools]
    result = _maybe_crew(execute, step, agent_id, tool_ids, query_text)
    return agent_id, tool_ids, 0.0, result


def _run_incontext(query: str, execute: bool, step):
    from src.baselines.frontier_incontext import incontext_route

    with step("normalize") as detail:
        query_text = normalize_query(query)
        detail["query_text"] = query_text
    with step("incontext_route") as detail:
        selection = incontext_route(query_text)
        detail["agent_id"] = selection["agent_id"]
        detail["tools"] = [{"id": t} for t in selection["tool_ids"]]
        detail["prompt_tokens"] = selection.get("prompt_tokens")
        detail["completion_tokens"] = selection.get("completion_tokens")
    agent_id = selection["agent_id"]
    tool_ids = selection["tool_ids"]
    cost = estimate_cost(selection.get("prompt_tokens"),
                         selection.get("completion_tokens"))
    result = _maybe_crew(execute, step, agent_id, tool_ids, query_text)
    return agent_id, tool_ids, cost, result


def _run_rag(query: str, execute: bool, step):
    from src.baselines.embeddings import agent_of_tool
    from src.baselines.rag_router import rag_retrieve
    from src.crews.specialist import MAX_TOOLS

    with step("normalize") as detail:
        query_text = normalize_query(query)
        detail["query_text"] = query_text
    with step("rag_retrieve") as detail:
        retrieved = rag_retrieve(query_text)
        detail["retrieved"] = _pairs(retrieved)
    with step("filter_to_agent") as detail:
        agent_of = agent_of_tool()
        top_tool = retrieved[0][0]
        agent_id = agent_of[top_tool]
        tool_ids = [tid for tid, _ in retrieved
                    if agent_of[tid] == agent_id][:MAX_TOOLS]
        if top_tool not in tool_ids:
            tool_ids = [top_tool, *tool_ids][:MAX_TOOLS]
        detail["agent_id"] = agent_id
        detail["tools"] = [{"id": t} for t in tool_ids]
    result = _maybe_crew(execute, step, agent_id, tool_ids, query_text)
    return agent_id, tool_ids, 0.0, result


_RUNNERS = {
    "incontext": _run_incontext,
    "rag": _run_rag,
    "probe": _run_probe,
}


def _run_arm(arm: str, query: str, execute: bool, emit_event: Callable) -> None:
    """Run one arm to completion; always emits exactly one arm_done event."""
    t0 = time.perf_counter()

    def emit_step(step_id, status, detail, t_ms):
        emit_event({"kind": "step", "arm": arm, "step_id": step_id,
                    "status": status, "t_ms": t_ms, "detail": detail})

    def step(step_id):
        return _step(emit_step, step_id)

    ok, agent_id, tool_ids, cost, result = True, "", [], 0.0, ""
    try:
        agent_id, tool_ids, cost, result = _RUNNERS[arm](query, execute, step)
    except Exception as exc:  # noqa: BLE001 — one arm failing must not kill others
        ok = False
        result = f"error: {exc}"

    emit_event({
        "kind": "arm_done", "arm": arm, "ok": ok, "latency_ms": _ms(t0),
        "cost_usd": cost, "agent_id": agent_id,
        "tool_id": tool_ids[0] if tool_ids else "",
        "tool_shortlist": tool_ids, "result": result[:MAX_RESULT_CHARS],
    })


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def event_stream(query: str, arms: list[str], execute: bool):
    """Async generator of SSE lines; runs each arm in its own worker thread."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    run_id = uuid.uuid4().hex[:12]

    def emit_event(event: dict) -> None:
        event["run_id"] = run_id
        loop.call_soon_threadsafe(queue.put_nowait, event)

    for arm in arms:
        asyncio.create_task(asyncio.to_thread(_run_arm, arm, query, execute, emit_event))

    yield _sse({"kind": "run_started", "run_id": run_id, "arms": arms,
                "execute": execute})

    finished = 0
    while finished < len(arms):
        event = await queue.get()
        yield _sse(event)
        if event.get("kind") == "arm_done":
            finished += 1

    yield _sse({"kind": "run_done", "run_id": run_id})
