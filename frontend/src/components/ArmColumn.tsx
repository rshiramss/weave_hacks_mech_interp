import { FlowGraph } from "./FlowGraph";
import { ProbeBank } from "./ProbeBank";
import type { ArmFlow, ArmRunState } from "../lib/types";

interface ArmColumnProps {
  arm: string;
  flow: ArmFlow;
  run?: ArmRunState;
  execute: boolean;
  accent: string;
  agents: Record<string, string>;
  weaveProject?: string;
}

const STATUS_LABEL: Record<string, string> = {
  idle: "idle",
  running: "running…",
  done: "done",
  error: "error",
};

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 10, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: 0.5 }}>
        {label}
      </span>
      <span style={{ fontSize: 13, color: "var(--text)", fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function weaveUrl(project: string): string {
  return `https://wandb.ai/${project}/weave`;
}

export function ArmColumn({ arm, flow, run, execute, accent, agents, weaveProject }: ArmColumnProps) {
  const status = run?.status ?? "idle";
  const summary = run?.summary;
  const latency = summary
    ? summary.latency_ms >= 1000
      ? `${(summary.latency_ms / 1000).toFixed(1)}s`
      : `${Math.round(summary.latency_ms)}ms`
    : "—";

  const agent = run?.chosenAgent;
  const specialty = agent ? agents[agent] : undefined;

  return (
    <section
      style={{
        display: "flex",
        flexDirection: "column",
        background: "var(--bg-panel)",
        border: "1px solid var(--border)",
        borderTop: `3px solid ${accent}`,
        borderRadius: "var(--radius)",
        overflow: "hidden",
        minWidth: 0,
      }}
    >
      <header style={{ padding: "12px 14px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontWeight: 700, color: "var(--text)", fontSize: 14 }}>{flow.label}</span>
        <span style={{ fontSize: 11, color: status === "error" ? "var(--status-error)" : "var(--text-dim)" }}>
          {STATUS_LABEL[status]}
        </span>
      </header>

      <div style={{ height: 520, borderTop: "1px solid var(--border)", borderBottom: "1px solid var(--border)" }}>
        <FlowGraph flow={flow} run={run} execute={execute} accent={accent} agents={agents} />
      </div>

      {arm === "probe" && (
        <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--border)" }}>
          <ProbeBank agents={agents} run={run} accent={accent} />
        </div>
      )}

      <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", gap: 18 }}>
          <MiniMetric label="latency" value={latency} />
          <MiniMetric label="cost" value={summary ? `$${summary.cost_usd.toFixed(5)}` : "—"} />
          <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
            <span style={{ fontSize: 10, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: 0.5 }}>
              routed agent
            </span>
            <span style={{ fontSize: 13, color: agent ? "var(--text)" : "var(--text-dim)", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {agent ? `${agent}${summary?.tool_id ? ` / ${summary.tool_id}` : ""}` : "—"}
            </span>
            {specialty && (
              <span style={{ fontSize: 11, color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {specialty}
              </span>
            )}
          </div>
        </div>

        {weaveProject && (
          <a
            href={weaveUrl(weaveProject)}
            target="_blank"
            rel="noreferrer"
            style={{ fontSize: 11, color: accent, fontWeight: 600, textDecoration: "none", alignSelf: "flex-start" }}
          >
            View in Weave →
          </a>
        )}
      </div>
    </section>
  );
}
