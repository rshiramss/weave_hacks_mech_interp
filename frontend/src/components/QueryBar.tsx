import { useState } from "react";

const SAMPLES = [
  "847 failed SSH logins from 203.0.113.44 on prod-bastion-01",
  "is msg-90144 a phishing email targeting our finance team",
  "WKSTN-042 is beaconing to 52.86.141.33 every 30 seconds",
  "pulled invoice.doc off PROXY-01, looks like it has macros",
];

interface QueryBarProps {
  onRun: (query: string, execute: boolean) => void;
  running: boolean;
}

export function QueryBar({ onRun, running }: QueryBarProps) {
  const [query, setQuery] = useState(SAMPLES[0]);
  const [execute, setExecute] = useState(false);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", gap: 10 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !running && query.trim() && onRun(query, execute)}
          placeholder="Describe a SOC alert…"
          style={{
            flex: 1,
            padding: "12px 14px",
            borderRadius: 10,
            border: "1px solid var(--border)",
            background: "var(--bg-panel)",
            color: "var(--text)",
            fontSize: 14,
            outline: "none",
          }}
        />
        <label style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text-dim)", fontSize: 13, whiteSpace: "nowrap" }}>
          <input type="checkbox" checked={execute} onChange={(e) => setExecute(e.target.checked)} />
          run crew
        </label>
        <button
          onClick={() => query.trim() && onRun(query, execute)}
          disabled={running || !query.trim()}
          style={{
            padding: "12px 22px",
            borderRadius: 10,
            border: "none",
            background: running ? "var(--bg-elevated)" : "var(--text)",
            color: running ? "var(--text-faint)" : "var(--bg)",
            fontWeight: 700,
            fontSize: 14,
            cursor: running ? "default" : "pointer",
          }}
        >
          {running ? "Routing…" : "Run ▶"}
        </button>
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {SAMPLES.map((s) => (
          <button
            key={s}
            onClick={() => setQuery(s)}
            style={{
              padding: "5px 10px",
              borderRadius: 999,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--text-faint)",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            {s.length > 46 ? s.slice(0, 46) + "…" : s}
          </button>
        ))}
      </div>
    </div>
  );
}
