// Shared types mirroring the FastAPI contracts (app/flow_spec.py, app/run_stream.py).

export type StepStatus = "pending" | "started" | "done" | "error";

export interface FlowNodeSpec {
  id: string;
  label: string;
  kind: "ingest" | "route" | "probe" | "agent" | "crew";
  execute_only?: boolean;
}

export interface ArmFlow {
  label: string;
  nodes: FlowNodeSpec[];
  edges: [string, string][];
}

export interface FlowSpec {
  arms: string[];
  flows: Record<string, ArmFlow>;
  // agent_id -> specialty description; powers the roster node + legend.
  agents?: Record<string, string>;
  // W&B/Weave project (entity/project) for the per-arm trace link.
  weave_project?: string;
}

export interface AgentCandidate {
  id: string;
  score: number;
}

export interface StepState {
  status: StepStatus;
  t_ms?: number | null;
  detail?: Record<string, unknown> | null;
}

export interface ArmSummary {
  ok: boolean;
  latency_ms: number;
  cost_usd: number;
  agent_id: string;
  tool_id: string;
  tool_shortlist: string[];
  result: string;
}

export type ArmStatus = "idle" | "running" | "done" | "error";

export interface ArmRunState {
  status: ArmStatus;
  steps: Record<string, StepState>;
  current?: string;
  summary?: ArmSummary;
  // Sub-agent the arm routed to, surfaced live from routing step detail.
  chosenAgent?: string;
  // Probe arm only: the agent_probe top-2 candidates with scores.
  agentCandidates?: AgentCandidate[];
}

export interface SSEEvent {
  kind: "run_started" | "step" | "arm_done" | "run_done";
  run_id: string;
  arm?: string;
  step_id?: string;
  status?: StepStatus;
  t_ms?: number | null;
  detail?: Record<string, unknown> | null;
  ok?: boolean;
  latency_ms?: number;
  cost_usd?: number;
  agent_id?: string;
  tool_id?: string;
  tool_shortlist?: string[];
  result?: string;
  arms?: string[];
  execute?: boolean;
}

// Per-arm aggregate metrics from /metrics (app/metrics.py).
export interface MetricsResponse {
  source?: string;
  run?: string;
  error?: string;
  arms: Record<string, Record<string, number>>;
}

export const ARM_ACCENT: Record<string, string> = {
  incontext: "var(--arm-incontext)",
  rag: "var(--arm-rag)",
  probe: "var(--arm-probe)",
};
