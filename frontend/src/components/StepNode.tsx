import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { StepState } from "../lib/types";

export type StepNodeData = {
  label: string;
  kind: string;
  accent: string;
  state?: StepState;
  current?: boolean;
};
type StepNodeType = Node<StepNodeData>;

const STATUS_COLOR: Record<string, string> = {
  pending: "var(--status-pending)",
  started: "var(--status-running)",
  done: "var(--status-done)",
  error: "var(--status-error)",
};

function summarizeDetail(detail?: Record<string, unknown> | null): string | null {
  if (!detail) return null;
  if (typeof detail.chosen === "string") return `→ ${detail.chosen}`;
  const cands = detail.candidates as { id: string; score: number }[] | undefined;
  if (cands?.length) {
    return cands
      .slice(0, 2)
      .map((c) => `${c.id} ${c.score.toFixed(2)}`)
      .join(" · ");
  }
  const tools = detail.tools as { id: string; score?: number }[] | undefined;
  if (tools?.length) {
    const top = tools[0];
    const topScore = top.score != null ? ` ${top.score.toFixed(2)}` : "";
    // Per-agent tool probe: name which of the 5 probes fired and its pool size.
    if (typeof detail.probe_agent === "string") {
      const pool = typeof detail.pool_size === "number" ? detail.pool_size : 40;
      return `${detail.probe_agent} · ${pool}→5 · ${top.id}${topScore}`;
    }
    const extra = tools.length - 1;
    return `${top.id}${topScore}` + (extra > 0 ? ` +${extra}` : "");
  }
  if (detail.prompt_tokens != null) {
    return `${detail.prompt_tokens}+${detail.completion_tokens ?? 0} tok`;
  }
  if (typeof detail.agent_id === "string") return `→ ${detail.agent_id}`;
  return null;
}

function StepNodeImpl({ data }: NodeProps<StepNodeType>) {
  const status = data.state?.status ?? "pending";
  const color = STATUS_COLOR[status];
  const running = data.current && status === "started";
  const detailLine = summarizeDetail(data.state?.detail);
  const tMs = data.state?.t_ms;

  return (
    <div
      style={{
        width: 220,
        boxSizing: "border-box",
        padding: "10px 12px",
        borderRadius: 10,
        background: "var(--bg-elevated)",
        border: `1px solid ${running ? data.accent : "var(--border)"}`,
        boxShadow: running ? `0 0 0 3px color-mix(in oklch, ${data.accent} 30%, transparent)` : "none",
        opacity: status === "pending" ? 0.55 : 1,
        transition: "opacity var(--duration-normal), box-shadow var(--duration-normal)",
      }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 9,
            height: 9,
            borderRadius: "50%",
            background: color,
            flexShrink: 0,
            animation: running ? "pulse 1.1s ease-in-out infinite" : "none",
          }}
        />
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)", lineHeight: 1.2 }}>
          {data.label}
        </span>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, gap: 8 }}>
        <span style={{ fontSize: 11, color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {detailLine ?? " "}
        </span>
        {tMs != null && (
          <span style={{ fontSize: 11, color: "var(--text-faint)", flexShrink: 0 }}>
            {tMs >= 1000 ? `${(tMs / 1000).toFixed(1)}s` : `${Math.round(tMs)}ms`}
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  );
}

export const StepNode = memo(StepNodeImpl);
