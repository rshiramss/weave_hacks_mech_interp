"""Log per-query routing traces to W&B using cached probe data (no forward passes).

Runs the orchestrator (agent) and tool probes over a 2000-query sample of the
cached layer-24 hidden states, then logs a per-query trace table plus hit-rate
summaries to a single W&B run.

Run from the project root:
    python scripts/log_traces.py
"""

import os
import json
import random
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import joblib
import wandb
from dotenv import load_dotenv

CACHE_PATH = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy"
QUERIES_PATH = "data/queries_mixed.json"
REGISTRY_PATH = "data/registry.json"

TOOL_PROBE_PATHS = ["data/probes/tool_probe_best.pkl", "data/probes/tool_probe.pkl"]
ORCH_PROBE_PATHS = [
    "data/probes/orchestrator_probe_best.pkl",
    "data/probes/orchestrator_probe.pkl",
]

N_SAMPLES = 2000
SEED = 42
TOPK = 5


# ---------------------------------------------------------------------------
# W&B credentials (same load_dotenv() / WANDB_API_KEY pattern as
# probe_ab_test.py's _wandb_setup()).
# ---------------------------------------------------------------------------

_WANDB_READY = False
WANDB_ENTITY = None
WANDB_PROJECT = None


def _wandb_setup():
    """Load credentials and log in once. Returns (entity, project)."""
    global _WANDB_READY, WANDB_ENTITY, WANDB_PROJECT
    if _WANDB_READY:
        return WANDB_ENTITY, WANDB_PROJECT

    load_dotenv()
    key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
    if not key:
        raise KeyError("Set WANDB_KEY (or WANDB_API_KEY) in your environment/.env")
    os.environ["WANDB_API_KEY"] = key

    WANDB_ENTITY = os.environ.get(
        "WANDB_ENTITY", "abrahambhatti525-santa-clara-university"
    )
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ToolOptim")
    wandb.login(key=key)
    _WANDB_READY = True
    return WANDB_ENTITY, WANDB_PROJECT


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_probe(paths):
    for path in paths:
        if os.path.exists(path):
            print(f"Loading probe from {path}")
            return joblib.load(path)
    raise FileNotFoundError(f"None of these probes exist: {paths}")


def load_agent_names():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    return {a["agent_id"]: a["name"] for a in registry["agents"]}


def load_sample():
    X = np.load(CACHE_PATH)
    with open(QUERIES_PATH) as f:
        rows = json.load(f)
    if X.shape[0] != len(rows):
        raise ValueError(f"cache rows ({X.shape[0]}) != query rows ({len(rows)}).")

    rng = random.Random(SEED)
    n = min(N_SAMPLES, len(rows))
    idx = rng.sample(range(len(rows)), n)
    queries = [rows[i]["query_text"] for i in idx]
    gt_tools = [rows[i]["tool_id"] for i in idx]
    gt_agents = [rows[i]["agent_id"] for i in idx]
    return X[idx], queries, gt_tools, gt_agents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COLUMNS = [
    "Query", "Predicted-Agent", "Agent-Confidence",
    "Tool-1", "Tool-1-Score",
    "Tool-2", "Tool-2-Score",
    "Tool-3", "Tool-3-Score",
    "Tool-4", "Tool-4-Score",
    "Tool-5", "Tool-5-Score",
    "Ground-Truth-Tool", "Ground-Truth-Agent", "Hit",
]


def main():
    tool_probe = load_probe(TOOL_PROBE_PATHS)
    orch_probe = load_probe(ORCH_PROBE_PATHS)
    agent_names = load_agent_names()

    X, queries, gt_tools, gt_agents = load_sample()
    n = len(queries)
    print(f"Scoring {n} queries (sampled, seed={SEED}) ...")

    orch_proba = orch_probe.predict_proba(X)
    orch_classes = orch_probe.classes_
    tool_proba = tool_probe.predict_proba(X)
    tool_classes = tool_probe.classes_
    top5_idx = np.argsort(-tool_proba, axis=1)[:, :TOPK]

    table_rows = []
    hits = 0
    per_agent = {}  # agent_id -> [hits, total]

    for i in range(n):
        a_idx = int(np.argmax(orch_proba[i]))
        pred_agent = orch_classes[a_idx]
        agent_conf = float(orch_proba[i, a_idx])

        top_tools = [tool_classes[j] for j in top5_idx[i]]
        top_scores = [float(tool_proba[i, j]) for j in top5_idx[i]]

        hit = gt_tools[i] in top_tools
        hits += hit
        bucket = per_agent.setdefault(gt_agents[i], [0, 0])
        bucket[0] += int(hit)
        bucket[1] += 1

        row = [queries[i], pred_agent, agent_conf]
        for t, s in zip(top_tools, top_scores):
            row.extend([t, s])
        row.extend([gt_tools[i], gt_agents[i], "✅" if hit else "❌"])
        table_rows.append(row)

        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}")

    overall_hit_rate = hits / n
    agent_hit_rate = {a: c[0] / c[1] for a, c in per_agent.items()}
    print(f"\nOverall hit rate (recall@{TOPK}): {overall_hit_rate:.4f}")

    try:
        entity, project = _wandb_setup()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            entity=entity,
            project=project,
            name=f"routing-traces-{ts}",
            group="traces",
            tags=["traces", "routing", "soc"],
            config={
                "total_queries": n,
                "hit_rate": overall_hit_rate,
                "per_agent_hit_rate": {
                    agent_names.get(a, a): r for a, r in agent_hit_rate.items()
                },
            },
            reinit=True,
        )

        table = wandb.Table(columns=COLUMNS)
        for row in table_rows:
            table.add_data(*row)

        bar_table = wandb.Table(
            data=[[agent_names.get(a, a), r] for a, r in agent_hit_rate.items()],
            columns=["Agent", "Hit-Rate"],
        )
        wandb.log({
            "routing_traces": table,
            "hit_rate_by_agent": wandb.plot.bar(
                bar_table, "Agent", "Hit-Rate", title="Hit Rate by Agent"
            ),
        })
        run.summary.update({"overall_hit_rate": overall_hit_rate})
        wandb.finish()
        print(f"Logged W&B run 'routing-traces-{ts}'.")
    except Exception as e:
        print(f"  [warn] W&B logging for routing traces failed: {e}")


if __name__ == "__main__":
    main()
