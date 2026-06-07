# Probe-Routed SOC Triage

WeaveHacks 4 — a probe-routed multi-agent system for security-operations (SOC)
alert triage. A small model + linear probes route an alert to the right
specialist **agent + tools**, faster and cheaper than a frontier model reasoning
over the whole 200-tool catalog. The system **routes and triages** — it does not
detect, block, or remediate attacks.

See [`agent_architecture.md`](agent_architecture.md) for the full design.

## What's here

| Layer | Where |
|-------|-------|
| 5 specialist CrewAI crews (mock tools, ≤5 per crew) | `src/crews/`, `src/tools/` |
| Baseline routers — Arm 1 in-context, Arm 2 RAG | `src/baselines/` |
| Hybrid compute — Modal (W&B Inference + GPU embeddings) | `modal_app/` |
| Routing eval — scorers, Weave Evaluation, wandb.Table | `eval/`, `src/routers.py`, `scripts/run_eval.py` |
| Demo — backend dispatcher, FastAPI, CLI | `src/demo_pipeline.py`, `app/server.py`, `scripts/demo_ui.py` |
| Probe router (Arm 3) | `src/probes/`, `src/routing/`, `src/flow.py`, `scripts/run_probe_router.py` |

## Setup

```bash
pip install crewai weave scikit-learn joblib pyyaml python-dotenv litellm \
            sentence-transformers modal fastapi uvicorn numpy
cp .env.example .env          # add WANDB_API_KEY
ollama pull qwen2.5:7b-instruct   # local routing + crew execution model
```

Set `WEAVE_PROJECT=<your-entity>/soc-probe-router` so traces land in your entity.

## One-command demo

```bash
python scripts/demo_ui.py                  # preset SSH alert, stub backend (instant)
python scripts/demo_ui.py --backend rag    # Arm 2 (local MiniLM embeddings, no setup)
python scripts/demo_ui.py --backend incontext   # Arm 1 (local Qwen, full catalog)
```

Pick the router with `ROUTER_BACKEND=stub|rag|incontext|probe` (env) or `--backend`.

Web endpoint:

```bash
uvicorn app.server:app --reload
curl -s localhost:8000/triage -H 'content-type: application/json' \
  -d '{"event": {"summary":"847 failed SSH logins","source_ip":"203.0.113.44","host":"prod-bastion-01"}}'
```

## Benchmark + observability (Weave)

```bash
modal deploy modal_app/incontext_route.py            # Arm 1 LLM (W&B Inference)
# Arm 2 RAG needs no precompute — local all-MiniLM-L6-v2 over data/registry.json

python scripts/run_baseline.py --arm rag --query "..."        # single route + trace
python scripts/publish_holdout.py                             # publish eval dataset
python scripts/run_eval.py --arms incontext,rag              # 3-arm Evaluation + Table
python scripts/demo_comparison.py --query "..."             # Trace Comparison demo
```

- **Traces / Evaluations / Leaderboard:** Weave project `soc-probe-router`.
- **Leaderboard:** Weave → Evaluations → filter `soc-routing-benchmark` →
  Visualize → Configure (`routing_exact_match` higher-better, `estimated_cost`
  lower-better) → save `soc-routing-leaderboard`.

## Status

| Stage | State |
|-------|-------|
| 3 Mock tools + specialist crews | done |
| 5 Baseline arms 1 & 2 (hybrid Modal + local) | done |
| 6 Holdout + Evaluation + wandb.Table | code done; needs `data/eval_holdout.jsonl` |
| 7 Leaderboard + cost + Trace Comparison | cost + comparison done; leaderboard = UI step |
| 8 Demo UI | done (stub/rag/incontext backends) |
| 2/4 Probes + ProbeRoutedFlow (Arm 3) | done — run with `ROUTER_BACKEND=probe` (retrain: `scripts/{audit_registry_ground,train_probes}.py`) |
