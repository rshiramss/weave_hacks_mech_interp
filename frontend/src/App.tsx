import { useEffect, useState } from "react";
import { fetchFlowSpec } from "./lib/api";
import { useRunStream } from "./hooks/useRunStream";
import { useEvalMetrics } from "./hooks/useEvalMetrics";
import { QueryBar } from "./components/QueryBar";
import { ArmColumn } from "./components/ArmColumn";
import { AgentLegend } from "./components/AgentLegend";
import { MetricsPanel } from "./components/MetricsPanel";
import { ARM_ACCENT, type FlowSpec } from "./lib/types";

const ARM_ORDER = ["incontext", "rag", "probe"];

export default function App() {
  const [spec, setSpec] = useState<FlowSpec | null>(null);
  const [execute, setExecute] = useState(false);
  const { arms, running, start } = useRunStream();
  const { metrics, loading, reload } = useEvalMetrics();

  useEffect(() => {
    fetchFlowSpec().then(setSpec).catch(() => setSpec(null));
  }, []);

  const handleRun = (query: string, exec: boolean) => {
    setExecute(exec);
    start(query, ARM_ORDER, exec);
  };

  const armList = spec ? ARM_ORDER.filter((a) => spec.flows[a]) : [];
  const agents = spec?.agents ?? {};

  return (
    <div
      style={{
        minHeight: "100%",
        padding: "24px clamp(16px, 3vw, 40px)",
        maxWidth: 1500,
        margin: "0 auto",
        display: "flex",
        flexDirection: "column",
        gap: 20,
      }}
    >
      <header>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, letterSpacing: -0.5, color: "var(--text)" }}>
          SOC Router · Live Flow
        </h1>
        <p style={{ color: "var(--text-dim)", margin: "4px 0 0", fontSize: 14 }}>
          One alert, three routers — watched step by step. Probe vs RAG vs full-catalog in-context.
        </p>
      </header>

      <QueryBar onRun={handleRun} running={running} />

      {spec?.agents && <AgentLegend agents={agents} />}

      <div className="arm-grid">
        {armList.map((arm) => (
          <ArmColumn
            key={arm}
            arm={arm}
            flow={spec!.flows[arm]}
            run={arms[arm]}
            execute={execute}
            accent={ARM_ACCENT[arm]}
            agents={agents}
            weaveProject={spec!.weave_project}
          />
        ))}
      </div>

      <MetricsPanel metrics={metrics} loading={loading} onReload={reload} />
    </div>
  );
}
