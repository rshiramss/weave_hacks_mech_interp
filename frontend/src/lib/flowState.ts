// Pure reducer: fold an SSE event into the per-arm run state.

import type { AgentCandidate, ArmRunState, SSEEvent } from "./types";

type ArmMap = Record<string, ArmRunState>;

function emptyArm(): ArmRunState {
  return { status: "running", steps: {} };
}

// The routed sub-agent and its candidates ride inside routing step detail:
//   probe agent_picker -> detail.chosen; agent_probe -> detail.candidates
//   incontext_route / rag filter_to_agent -> detail.agent_id
function agentFromDetail(detail?: Record<string, unknown> | null): string | undefined {
  if (!detail) return undefined;
  if (typeof detail.chosen === "string") return detail.chosen;
  if (typeof detail.agent_id === "string") return detail.agent_id;
  return undefined;
}

function candidatesFromDetail(
  detail?: Record<string, unknown> | null,
): AgentCandidate[] | undefined {
  const cands = detail?.candidates;
  if (!Array.isArray(cands)) return undefined;
  return cands as AgentCandidate[];
}

export function reduceEvent(prev: ArmMap, ev: SSEEvent): ArmMap {
  if (ev.kind === "step" && ev.arm && ev.step_id) {
    const arm = prev[ev.arm] ?? emptyArm();
    const steps = {
      ...arm.steps,
      [ev.step_id]: {
        status: ev.status ?? "started",
        t_ms: ev.t_ms,
        detail: ev.detail,
      },
    };
    // current = the step that just started; cleared when it finishes.
    let current = arm.current;
    if (ev.status === "started") current = ev.step_id;
    else if (arm.current === ev.step_id) current = undefined;

    const chosenAgent = agentFromDetail(ev.detail) ?? arm.chosenAgent;
    const agentCandidates = candidatesFromDetail(ev.detail) ?? arm.agentCandidates;
    return {
      ...prev,
      [ev.arm]: { ...arm, steps, current, chosenAgent, agentCandidates },
    };
  }

  if (ev.kind === "arm_done" && ev.arm) {
    const arm = prev[ev.arm] ?? emptyArm();
    return {
      ...prev,
      [ev.arm]: {
        ...arm,
        status: ev.ok ? "done" : "error",
        current: undefined,
        chosenAgent: arm.chosenAgent ?? ev.agent_id,
        summary: {
          ok: !!ev.ok,
          latency_ms: ev.latency_ms ?? 0,
          cost_usd: ev.cost_usd ?? 0,
          agent_id: ev.agent_id ?? "",
          tool_id: ev.tool_id ?? "",
          tool_shortlist: ev.tool_shortlist ?? [],
          result: ev.result ?? "",
        },
      },
    };
  }

  return prev;
}
