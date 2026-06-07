interface AgentLegendProps {
  agents: Record<string, string>;
}

// Always-visible catalog of the five specialist sub-agents the routers choose
// among, so the roster reads at a glance before any run.
export function AgentLegend({ agents }: AgentLegendProps) {
  const entries = Object.entries(agents);
  if (entries.length === 0) return null;

  return (
    <section
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: 10,
        padding: "10px 14px",
        background: "var(--bg-panel)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
      }}
    >
      <span
        style={{
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: 0.5,
          textTransform: "uppercase",
          color: "var(--text-faint)",
          marginRight: 4,
        }}
      >
        Specialist sub-agents
      </span>
      {entries.map(([id, specialty]) => (
        <div
          key={id}
          title={specialty}
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 1,
            padding: "5px 10px",
            background: "var(--bg-elevated)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            minWidth: 0,
          }}
        >
          <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{id}</span>
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>{specialty}</span>
        </div>
      ))}
    </section>
  );
}
