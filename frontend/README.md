# SOC Router · Live Flow (front end)

React + Vite + TypeScript + React Flow UI that runs one SOC alert through all
three router arms (in-context / RAG / probe) and shows each as a live,
hierarchical step graph, with aggregate eval metrics pulled from W&B.

Implements [`../frontend_implementation.md`](../frontend_implementation.md).

## Run

Backend (repo root) — needs `WANDB_API_KEY`, the deployed Modal extractor, and
local Ollama running:

```bash
uvicorn app.server:app --port 8000
```

Front end:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api/* -> :8000)
```

Type-check + production build: `npm run build`.

## What it shows

- **Three columns** (in-context / RAG / probe), each a top-to-bottom step graph
  from `GET /flow-spec`. Nodes light up live as `GET /triage/stream` (SSE) emits
  per-step events; the running step is highlighted and pulses. Toggle **run crew**
  to include the crew-execution nodes.
- **Per-run** latency / cost / chosen route under each graph (from the live run).
- **Holdout eval · W&B aggregate** table from `GET /metrics` — routing-exact,
  agent recall@2, tool recall@5, mean latency, mean cost per arm. Accuracy is the
  leak-free holdout aggregate produced by `scripts/run_eval.py`; `↻ refresh`
  re-pulls.

## Layout

```
src/
  App.tsx                  # page: query bar + 3 arm columns + metrics panel
  components/
    QueryBar.tsx           # query input, sample chips, run-crew toggle
    ArmColumn.tsx          # one arm: header + graph + per-run metrics
    FlowGraph.tsx          # React Flow graph (dagre layout)
    StepNode.tsx           # custom node: status, timing, step detail
    MetricsPanel.tsx       # W&B aggregate comparison table
  hooks/
    useRunStream.ts        # SSE -> per-arm step state
    useEvalMetrics.ts      # /metrics fetch + refresh
  lib/
    types.ts  api.ts  flowState.ts (SSE reducer)  layout.ts (dagre)
```
