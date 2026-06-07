import { ARM_ACCENT, type MetricsResponse } from "../lib/types";

const ROWS: { key: string; label: string; fmt: (v: number) => string }[] = [
  { key: "routing_exact_match", label: "Routing exact", fmt: (v) => `${(v * 100).toFixed(0)}%` },
  { key: "agent_recall_at_2", label: "Agent recall@2", fmt: (v) => `${(v * 100).toFixed(0)}%` },
  { key: "tool_recall_at_5", label: "Tool recall@5", fmt: (v) => `${(v * 100).toFixed(0)}%` },
  { key: "latency_ms", label: "Latency (mean)", fmt: (v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`) },
  { key: "cost_usd", label: "Cost (mean)", fmt: (v) => `$${v.toFixed(5)}` },
];

const ARM_ORDER = ["incontext", "rag", "probe"];

interface MetricsPanelProps {
  metrics: MetricsResponse | null;
  loading: boolean;
  onReload: () => void;
}

export function MetricsPanel({ metrics, loading, onReload }: MetricsPanelProps) {
  const arms = ARM_ORDER.filter((a) => metrics?.arms?.[a]);
  const empty = arms.length === 0;
  const cell = (v: number | undefined, fmt: (n: number) => string) =>
    v == null ? "—" : fmt(v);

  return (
    <section
      style={{
        background: "var(--bg-panel)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: "16px 18px",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <h2 style={{ margin: 0, fontSize: 15, color: "var(--text)" }}>
          Holdout eval · W&amp;B aggregate
        </h2>
        <button
          onClick={onReload}
          disabled={loading}
          style={{
            background: "transparent",
            border: "1px solid var(--border)",
            color: "var(--text-dim)",
            borderRadius: 8,
            padding: "4px 10px",
            fontSize: 12,
            cursor: loading ? "default" : "pointer",
          }}
        >
          {loading ? "…" : "↻ refresh"}
        </button>
      </div>

      {empty ? (
        <p style={{ color: "var(--text-faint)", fontSize: 13, margin: 0 }}>
          No eval found. Run <code>scripts/run_eval.py</code> to populate.
          {metrics?.error ? ` (${metrics.error})` : ""}
        </p>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", color: "var(--text-faint)", fontWeight: 500, padding: "4px 8px" }}>
                metric
              </th>
              {arms.map((a) => (
                <th key={a} style={{ textAlign: "right", padding: "4px 8px", color: ARM_ACCENT[a], fontWeight: 700 }}>
                  {a}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row.key} style={{ borderTop: "1px solid var(--border)" }}>
                <td style={{ padding: "6px 8px", color: "var(--text-dim)" }}>{row.label}</td>
                {arms.map((a) => (
                  <td key={a} style={{ padding: "6px 8px", textAlign: "right", color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>
                    {cell(metrics!.arms[a][row.key], row.fmt)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <p style={{ color: "var(--text-faint)", fontSize: 11, marginTop: 12, marginBottom: 0 }}>
        Accuracy = aggregate over the leak-free holdout
        {metrics?.run ? ` · ${metrics.run}` : ""}
        {metrics?.source === "local" ? " · local cache" : ""}. Per-run latency/cost
        shown live above per arm.
      </p>
    </section>
  );
}
