import { memo } from "react";
import type { ArmRunState } from "../lib/types";

interface ToolScore {
  id: string;
  score?: number;
}

interface ProbeBankProps {
  agents: Record<string, string>; // agent_id -> specialty (the 5-agent roster)
  run?: ArmRunState;
  accent: string;
}

function toolsFromRun(run?: ArmRunState): ToolScore[] {
  const detail = run?.steps?.tool_probe?.detail as
    | { tools?: ToolScore[] }
    | undefined;
  return detail?.tools ?? [];
}

/**
 * Tells the per-agent-probe story: five probes, exactly one fires. Renders the
 * roster as a row of chips; once agent_picker resolves, the chosen probe lights
 * and the other four dim; the lit probe expands to its top-5 tools as score bars.
 * All data is already in run state — no new SSE contract.
 */
function ProbeBankImpl({ agents, run, accent }: ProbeBankProps) {
  const ids = Object.keys(agents);
  const chosen = run?.chosenAgent;
  const fired = run?.steps?.tool_probe?.status === "done";
  const tools = toolsFromRun(run);
  // Normalize bar width to the top score so a tight softmax over 40 classes still reads.
  const max = Math.max(...tools.map((t) => t.score ?? 0), 1e-6);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <span
        style={{
          fontSize: 10,
          color: "var(--text-faint)",
          textTransform: "uppercase",
          letterSpacing: 0.5,
        }}
      >
        per-agent tool probes · 1 of {ids.length} fires
      </span>
      <div style={{ display: "flex", gap: 6 }}>
        {ids.map((id) => {
          const active = id === chosen;
          return (
            <div
              key={id}
              title={agents[id]}
              style={{
                flex: 1,
                minWidth: 0,
                padding: "6px 8px",
                borderRadius: 8,
                border: `1px solid ${active ? accent : "var(--border)"}`,
                background: active
                  ? `color-mix(in oklch, ${accent} 14%, transparent)`
                  : "var(--bg-elevated)",
                opacity: !chosen ? 0.7 : active ? 1 : 0.35,
                transition:
                  "opacity var(--duration-normal), border-color var(--duration-normal)",
              }}
            >
              <span
                style={{
                  display: "block",
                  fontSize: 11,
                  fontWeight: 600,
                  color: "var(--text)",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {id}
              </span>
              <span style={{ fontSize: 9, color: "var(--text-faint)" }}>
                40 tools
              </span>
            </div>
          );
        })}
      </div>
      {fired && tools.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {tools.slice(0, 5).map((t) => (
            <div
              key={t.id}
              style={{ display: "flex", alignItems: "center", gap: 8 }}
            >
              <span
                style={{
                  width: 150,
                  fontSize: 11,
                  color: "var(--text-dim)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {t.id}
              </span>
              <div
                style={{
                  flex: 1,
                  height: 6,
                  borderRadius: 3,
                  background: "var(--bg-elevated)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${Math.round(((t.score ?? 0) / max) * 100)}%`,
                    height: "100%",
                    background: accent,
                    transition: "width var(--duration-normal)",
                  }}
                />
              </div>
              <span
                style={{
                  width: 36,
                  textAlign: "right",
                  fontSize: 10,
                  color: "var(--text-faint)",
                }}
              >
                {t.score != null ? t.score.toFixed(2) : ""}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export const ProbeBank = memo(ProbeBankImpl);
