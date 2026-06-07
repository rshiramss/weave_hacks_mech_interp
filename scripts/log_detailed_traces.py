"""Log rich per-query routing traces to W&B Weave + summary metrics to W&B.

Uses cached layer-24 hidden states only (no model loading). For each query in
the held-out test split (same 80/20 stratified split as train_probes.py,
seed=42), runs the orchestrator + tool probes and a MiniLM RAG baseline, then
logs a @weave.op trace per query plus summary tables/charts to a single W&B run.

Run from the project root:
    python scripts/log_detailed_traces.py
"""

import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import joblib
import weave
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

CACHE_PATH = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy"
QUERIES_PATH = "data/queries_mixed.json"
REGISTRY_PATH = "data/registry.json"
TOOL_PROBE_PATHS = {
    "log_search": "data/probes/tool_probe_log_search.pkl",
    "threat_intel": "data/probes/tool_probe_threat_intel.pkl",
    "malware_analysis": "data/probes/tool_probe_malware_analysis.pkl",
    "network_analysis": "data/probes/tool_probe_network_analysis.pkl",
    "email_security": "data/probes/tool_probe_email_security.pkl",
}
ORCH_PROBE_PATH = "data/probes/orchestrator_probe_best.pkl"
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEED = 42
TOPK = 10


# ---------------------------------------------------------------------------
# Credentials (same load_dotenv() / WANDB_KEY pattern as other scripts).
# ---------------------------------------------------------------------------

load_dotenv()
_key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
if not _key:
    raise KeyError("Set WANDB_KEY (or WANDB_API_KEY) in your environment/.env")
os.environ["WANDB_API_KEY"] = _key
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "abrahambhatti525-santa-clara-university")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ToolOptim")


# ---------------------------------------------------------------------------
# Load data, probes, registry, embedder
# ---------------------------------------------------------------------------

print("Loading cache, queries, probes, registry, embedder...")
X = np.load(CACHE_PATH)
with open(QUERIES_PATH) as f:
    rows = json.load(f)
if X.shape[0] != len(rows):
    raise ValueError(f"cache rows ({X.shape[0]}) != query rows ({len(rows)}).")

y_tool = np.array([r["tool_id"] for r in rows])
y_agent = np.array([r["agent_id"] for r in rows])

idx = np.arange(len(rows))
_, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_tool)
print(f"  test set: {len(test_idx)} queries")

X_te = X[test_idx]
yt_te = y_tool[test_idx]
ya_te = y_agent[test_idx]
queries_te = [rows[i]["query_text"] for i in test_idx]

tool_probes = {agent: joblib.load(path) for agent, path in TOOL_PROBE_PATHS.items()}
orch_probe = joblib.load(ORCH_PROBE_PATH)

with open(REGISTRY_PATH) as f:
    _registry = json.load(f)
tool_desc = {t["tool_id"]: t["description"] for t in _registry["tools"]}
TOOL_IDS = [t["tool_id"] for t in _registry["tools"]]

from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer(EMBEDDER_NAME)
_tool_embeddings = _embedder.encode(
    [tool_desc[t] for t in TOOL_IDS], normalize_embeddings=True
)


# ---------------------------------------------------------------------------
# Weave: per-query trace op
# ---------------------------------------------------------------------------

weave.init(f"{WANDB_ENTITY}/{WANDB_PROJECT}")


@weave.op
def route_query_two_stage_best_vs_rag_minilm(query_text: str, ground_truth_tool: str,
                                             ground_truth_agent: str,
                                             output: dict) -> dict:
    return output


# ---------------------------------------------------------------------------
# Score the test set
# ---------------------------------------------------------------------------

print("Scoring probes + RAG on test set (two-stage: orchestrator -> per-agent tool probe)...")
orch_proba = orch_probe.predict_proba(X_te)
orch_classes = orch_probe.classes_
predicted_agents = orch_classes[np.argmax(orch_proba, axis=1)]
n = len(test_idx)

# Two-stage tool routing: batch each row through the tool probe owned by its
# *predicted* agent (not ground truth), grouping rows by predicted agent so we
# can call predict_proba in batches instead of one row at a time.
top_tools_per_row = [None] * n
for agent, probe in tool_probes.items():
    sel = np.where(predicted_agents == agent)[0]
    if len(sel) == 0:
        continue
    sub_proba = probe.predict_proba(X_te[sel])
    sub_classes = probe.classes_
    sub_order = np.argsort(-sub_proba, axis=1)[:, :10]
    for row_pos, row_idx in enumerate(sel):
        top_tools_per_row[row_idx] = [sub_classes[j] for j in sub_order[row_pos]]

q_embeddings = _embedder.encode(queries_te, normalize_embeddings=True)
rag_sims = q_embeddings @ _tool_embeddings.T  # (n_test, n_tools)

table_rows = []
rank_buckets = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "6-10": 0, "not_found": 0}
per_agent = {}  # agent_id -> dict of accumulators
agent_confidences = []
rag_recall5 = rag_recall10 = 0
rag_ranks = []
probe_ranks = []

for i in range(n):
    gt_tool = yt_te[i]
    gt_agent = ya_te[i]

    a_idx = int(np.argmax(orch_proba[i]))
    predicted_agent = orch_classes[a_idx]
    agent_confidence = float(orch_proba[i, a_idx])
    agent_confidences.append(agent_confidence)

    # Two-stage: tool ranking comes from the predicted agent's own tool probe
    # (only ~40 tools wide), so a wrongly-routed query can never surface the
    # ground-truth tool here — that's reflected in two_stage_tool_rank/hit below.
    top_tools = top_tools_per_row[i]
    top_5_tools = top_tools[:5]
    top_10_tools = top_tools
    predicted_tool = top_tools[0]

    tool_rank = top_tools.index(gt_tool) + 1 if gt_tool in top_tools else None
    probe_hit_5 = gt_tool in top_5_tools
    probe_hit_10 = gt_tool in top_10_tools
    agent_correct = predicted_agent == gt_agent
    agent_failure = probe_hit_10 and predicted_tool != gt_tool

    if tool_rank is None:
        rank_buckets["not_found"] += 1
    elif tool_rank <= 5:
        rank_buckets[tool_rank] += 1
        probe_ranks.append(tool_rank)
    else:
        rank_buckets["6-10"] += 1
        probe_ranks.append(tool_rank)

    rag_order = np.argsort(-rag_sims[i])[:10]
    rag_top_tools = [TOOL_IDS[j] for j in rag_order]
    rag_rank = rag_top_tools.index(gt_tool) + 1 if gt_tool in rag_top_tools else None
    rag_hit_5 = rag_rank is not None and rag_rank <= 5
    rag_hit_10 = rag_rank is not None
    if rag_hit_5:
        rag_recall5 += 1
    if rag_hit_10:
        rag_recall10 += 1
    if rag_rank is not None:
        rag_ranks.append(rag_rank)

    output = {
        "model": "Two-Stage-BEST-Probe-vs-RAG-MiniLM",
        "two_stage_predicted_agent": predicted_agent,
        "orchestrator_agent_confidence": agent_confidence,
        "two_stage_predicted_tool": predicted_tool,
        "two_stage_tool_rank": tool_rank,
        "two_stage_top_5_tools": top_5_tools,
        "two_stage_top_10_tools": top_10_tools,
        "two_stage_hit_5": probe_hit_5,
        "two_stage_hit_10": probe_hit_10,
        "orchestrator_agent_correct": agent_correct,
        "two_stage_agent_failure": agent_failure,
        "rag_minilm_tool_rank": rag_rank,
        "rag_minilm_hit_5": rag_hit_5,
        "query_style": "mixed",
    }

    try:
        route_query_two_stage_best_vs_rag_minilm(queries_te[i], gt_tool, gt_agent, output)
    except Exception as e:
        if i == 0:
            print(f"  [warn] weave trace logging failed: {e}")

    bucket = per_agent.setdefault(gt_agent, {
        "hit5": 0, "hit10": 0, "agent_correct": 0, "conf_sum": 0.0, "n": 0,
    })
    bucket["hit5"] += int(probe_hit_5)
    bucket["hit10"] += int(probe_hit_10)
    bucket["agent_correct"] += int(agent_correct)
    bucket["conf_sum"] += agent_confidence
    bucket["n"] += 1

    table_rows.append([
        queries_te[i], gt_tool, gt_agent,
        predicted_agent, agent_confidence,
        predicted_tool, tool_rank,
        top_5_tools, top_10_tools,
        probe_hit_5, probe_hit_10,
        agent_correct, agent_failure,
        rag_rank, rag_hit_5,
        "mixed",
    ])

    if (i + 1) % 100 == 0 or (i + 1) == n:
        print(f"  {i + 1}/{n}")

# ---------------------------------------------------------------------------
# Aggregate summary stats
# ---------------------------------------------------------------------------

overall_probe_recall5 = float(np.mean([r[9] for r in table_rows]))
overall_probe_recall10 = float(np.mean([r[10] for r in table_rows]))
overall_rag_recall5 = rag_recall5 / n
overall_rag_recall10 = rag_recall10 / n
avg_agent_confidence = float(np.mean(agent_confidences))
agent_accuracy = float(np.mean([r[11] for r in table_rows]))
agent_failure_rate = float(np.mean([r[12] for r in table_rows]))
pct_rank1 = rank_buckets[1] / n
pct_rank_top3 = (rank_buckets[1] + rank_buckets[2] + rank_buckets[3]) / n
avg_probe_rank = float(np.mean(probe_ranks)) if probe_ranks else None
avg_rag_rank = float(np.mean(rag_ranks)) if rag_ranks else None

print("\n==================== OVERALL ====================")
print(f"Two-Stage BEST Recall@5:     {overall_probe_recall5:.4f}")
print(f"Two-Stage BEST Recall@10:    {overall_probe_recall10:.4f}")
print(f"RAG MiniLM Recall@5:         {overall_rag_recall5:.4f}")
print(f"RAG MiniLM Recall@10:        {overall_rag_recall10:.4f}")
print(f"Orchestrator avg confidence: {avg_agent_confidence:.4f}")
print(f"Orchestrator agent accuracy: {agent_accuracy:.4f}")
print(f"Two-Stage agent failure rate:{agent_failure_rate:.4f}")
print(f"Two-Stage BEST %% rank 1:    {pct_rank1:.4f}")
print(f"Two-Stage BEST %% rank top-3:{pct_rank_top3:.4f}")

# ---------------------------------------------------------------------------
# Log to W&B workspace
# ---------------------------------------------------------------------------

try:
    wandb.login(key=_key)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"two-stage-best-vs-rag-minilm-traces-{ts}",
        group="two_stage_best_vs_rag_minilm",
        tags=[
            "two-stage-best",
            "orchestrator-probe-best",
            "per-agent-tool-probes",
            "rag-minilm",
            "weave-traces",
        ],
        reinit=True,
    )

    table_columns = [
        "query_text", "ground_truth_tool", "ground_truth_agent",
        "two_stage_predicted_agent", "orchestrator_agent_confidence",
        "two_stage_predicted_tool", "two_stage_tool_rank",
        "two_stage_top_5_tools", "two_stage_top_10_tools",
        "two_stage_hit_5", "two_stage_hit_10",
        "orchestrator_agent_correct", "two_stage_agent_failure",
        "rag_minilm_tool_rank", "rag_minilm_hit_5",
        "query_style",
    ]
    detail_table = wandb.Table(columns=table_columns)
    for row in table_rows:
        detail_table.add_data(*[
            json.dumps(v) if isinstance(v, list) else v for v in row
        ])

    # 1. Confidence distribution histogram
    confidence_hist = wandb.Histogram(agent_confidences)

    # 2. Tool rank distribution bar chart
    rank_labels = ["1", "2", "3", "4", "5", "6-10", "not_found"]
    rank_counts = [rank_buckets[1], rank_buckets[2], rank_buckets[3],
                   rank_buckets[4], rank_buckets[5], rank_buckets["6-10"],
                   rank_buckets["not_found"]]
    rank_table = wandb.Table(
        data=[[lbl, cnt] for lbl, cnt in zip(rank_labels, rank_counts)],
        columns=["rank", "count"],
    )

    # 3. Per-agent metrics table
    per_agent_two_stage_table = wandb.Table(
        columns=[
            "agent_id",
            "two_stage_best_recall@5",
            "two_stage_best_recall@10",
            "orchestrator_agent_accuracy",
            "orchestrator_avg_confidence",
        ]
    )
    for aid, b in sorted(per_agent.items()):
        per_agent_two_stage_table.add_data(
            aid,
            b["hit5"] / b["n"],
            b["hit10"] / b["n"],
            b["agent_correct"] / b["n"],
            b["conf_sum"] / b["n"],
        )

    # 4. Two-Stage BEST vs RAG MiniLM comparison table
    comparison_table = wandb.Table(columns=["metric", "two_stage_best", "rag_minilm"])
    comparison_table.add_data("Recall@5", overall_probe_recall5, overall_rag_recall5)
    comparison_table.add_data("Recall@10", overall_probe_recall10, overall_rag_recall10)
    comparison_table.add_data("Avg rank of correct tool", avg_probe_rank, avg_rag_rank)

    wandb.log({
        "two_stage_best/detailed_traces": detail_table,
        "orchestrator/agent_confidence_distribution": confidence_hist,
        "two_stage_best/tool_rank_distribution": wandb.plot.bar(
            rank_table, "rank", "count", title="Two-Stage BEST Tool Rank Distribution"
        ),
        "two_stage_best/per_agent_metrics": per_agent_two_stage_table,
        "two_stage_best_vs_rag_minilm/comparison": comparison_table,
    })

    run.summary.update({
        "two_stage_best/recall_at_5": overall_probe_recall5,
        "two_stage_best/recall_at_10": overall_probe_recall10,
        "rag_minilm/recall_at_5": overall_rag_recall5,
        "rag_minilm/recall_at_10": overall_rag_recall10,
        "orchestrator/avg_agent_confidence": avg_agent_confidence,
        "orchestrator/agent_accuracy": agent_accuracy,
        "two_stage_best/agent_failure_rate": agent_failure_rate,
        "two_stage_best/pct_rank1": pct_rank1,
        "two_stage_best/pct_rank_top3": pct_rank_top3,
    })

    wandb.finish()
    print(f"\nLogged W&B run 'two-stage-best-vs-rag-minilm-traces-{ts}'.")
except Exception as e:
    print(f"  [warn] W&B logging failed: {e}")

print("\nDone.")
