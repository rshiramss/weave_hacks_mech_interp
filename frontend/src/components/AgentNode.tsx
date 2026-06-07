import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { AgentCandidate } from "../lib/types";
import { AGENT_NODE_HEIGHT, NODE_WIDTH } from "../lib/layout";

// The routed-specialist node: shows the full roster of sub-agents and lights up
// the one this arm routed to. Fed live from the run's chosen agent.
export type AgentNodeData = {
  accent: string;
  agents: Record<string, string>;
  chosenAgent?: string;
  candidates?: AgentCandidate[];
};
type AgentNodeType = Node<AgentNodeData>;

function AgentRow({
  agentId,
  chosen,
  accent,
  score,
}: {
  agentId: string;
  chosen: boolean;
  accent: string;
  score?: number;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "3px 6px",
        borderRadius: 6,
        background: chosen ? `color-mix(in oklch, ${accent} 22%, transparent)` : "transparent",
        border: `1px solid ${chosen ? accent : "transparent"}`,
        transition: "background var(--duration-normal), border-color var(--duration-normal)",
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          flexShrink: 0,
          background: chosen ? accent : "var(--text-faint)",
        }}
      />
      <span
        style={{
          fontSize: 12,
          fontWeight: chosen ? 700 : 500,
          color: chosen ? "var(--text)" : "var(--text-dim)",
          opacity: chosen ? 1 : 0.7,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          flex: 1,
        }}
      >
        {chosen ? "→ " : ""}
        {agentId}
      </span>
      {score != null && (
        <span style={{ fontSize: 10, color: "var(--text-faint)", flexShrink: 0 }}>
          {score.toFixed(2)}
        </span>
      )}
    </div>
  );
}

function AgentNodeImpl({ data }: NodeProps<AgentNodeType>) {
  const { accent, agents, chosenAgent, candidates } = data;
  const agentIds = Object.keys(agents);
  const scoreOf = new Map((candidates ?? []).map((c) => [c.id, c.score]));
  const specialty = chosenAgent ? agents[chosenAgent] : undefined;

  return (
    <div
      style={{
        width: NODE_WIDTH,
        minHeight: AGENT_NODE_HEIGHT,
        boxSizing: "border-box",
        padding: "10px 12px",
        borderRadius: 10,
        background: "var(--bg-elevated)",
        border: `1px solid ${chosenAgent ? accent : "var(--border)"}`,
        boxShadow: chosenAgent
          ? `0 0 0 3px color-mix(in oklch, ${accent} 22%, transparent)`
          : "none",
        opacity: chosenAgent ? 1 : 0.7,
        transition: "opacity var(--duration-normal), box-shadow var(--duration-normal)",
      }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />

      <div
        style={{
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: 0.5,
          textTransform: "uppercase",
          color: "var(--text-faint)",
          marginBottom: 6,
        }}
      >
        Specialist sub-agent
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {agentIds.map((id) => (
          <AgentRow
            key={id}
            agentId={id}
            chosen={id === chosenAgent}
            accent={accent}
            score={scoreOf.get(id)}
          />
        ))}
      </div>

      <div
        style={{
          marginTop: 6,
          fontSize: 11,
          lineHeight: 1.25,
          color: chosenAgent ? "var(--text-dim)" : "var(--text-faint)",
          minHeight: 14,
        }}
      >
        {specialty ?? "awaiting routing…"}
      </div>

      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  );
}

export const AgentNode = memo(AgentNodeImpl);
