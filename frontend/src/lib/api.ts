import type { FlowSpec, MetricsResponse } from "./types";

export async function fetchFlowSpec(): Promise<FlowSpec> {
  const res = await fetch("/api/flow-spec");
  if (!res.ok) throw new Error(`flow-spec ${res.status}`);
  return res.json();
}

export async function fetchMetrics(refresh = false): Promise<MetricsResponse> {
  const res = await fetch(`/api/metrics${refresh ? "?refresh=1" : ""}`);
  if (!res.ok) throw new Error(`metrics ${res.status}`);
  return res.json();
}
