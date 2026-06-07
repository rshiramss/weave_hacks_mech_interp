"""Evaluate the saved tool + orchestrator probes (no training).

Replays the same 80/20 stratified split as scripts/train_probes.py (seed=42)
and reports tool-probe recall/top-1, orchestrator agent accuracy, and a
"probe had it, top-1 missed" failure rate, both overall and per agent.

CPU-only, fast (<30s). Just joblib + numpy + sklearn.

Run from the project root:
    python scripts/eval_probe.py
"""

import json
import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

CACHE_PATH = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy"
QUERIES_PATH = "data/queries_mixed.json"
CLEAN_INDICES_PATH = "data/clean_indices_mixed.json"
SEED = 42


def pick_probe(best, fallback):
    path = best if os.path.exists(best) else fallback
    if not os.path.exists(path):
        print(f"ERROR: neither {best} nor {fallback} found.")
        sys.exit(1)
    return joblib.load(path), path


def topk_labels(proba, classes, k):
    """Return an (n, k) array of class labels for the top-k columns per row."""
    k = min(k, proba.shape[1])
    order = np.argsort(-proba, axis=1)[:, :k]
    return classes[order]


def metric_block(recall5, recall10, top1, agent_ok, failure):
    """Aggregate the five boolean arrays into a dict of means."""
    return {
        "top1": float(np.mean(top1)),
        "recall5": float(np.mean(recall5)),
        "recall10": float(np.mean(recall10)),
        "agent_acc": float(np.mean(agent_ok)),
        "agent_failure": float(np.mean(failure)),
        "n": int(len(top1)),
    }


def main():
    print("Loading cache, labels, and probes...")
    X = np.load(CACHE_PATH)
    with open(QUERIES_PATH) as f:
        rows = json.load(f)

    # The cache is indexed by original row position; if the query file was
    # cleaned (fewer rows than the cache) realign by keeping only the original
    # rows recorded in the clean-indices mapping so X[i] matches rows[i].
    if X.shape[0] != len(rows):
        if os.path.exists(CLEAN_INDICES_PATH):
            with open(CLEAN_INDICES_PATH) as f:
                keep_idx = json.load(f)
            if len(keep_idx) == len(rows):
                X = X[keep_idx]
                print(f"  realigned cache {len(keep_idx)} rows via "
                      f"{CLEAN_INDICES_PATH}")
        if X.shape[0] != len(rows):
            print(f"ERROR: cache rows ({X.shape[0]}) != query rows "
                  f"({len(rows)}) and could not realign.")
            sys.exit(1)

    y_tool = np.array([r["tool_id"] for r in rows])
    y_agent = np.array([r["agent_id"] for r in rows])

    tool_probe, tool_path = pick_probe(
        "data/probes/tool_probe_best.pkl", "data/probes/tool_probe.pkl"
    )
    orch_probe, orch_path = pick_probe(
        "data/probes/orchestrator_probe_best.pkl",
        "data/probes/orchestrator_probe.pkl",
    )
    print(f"  tool probe:         {tool_path} ({len(tool_probe.classes_)} classes)")
    print(f"  orchestrator probe: {orch_path} ({len(orch_probe.classes_)} classes)")

    # Same split as scripts/train_probes.py: stratify by tool_id, seed=42.
    idx = np.arange(len(rows))
    _, test_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y_tool
    )
    X_te = X[test_idx]
    yt_te = y_tool[test_idx]
    ya_te = y_agent[test_idx]
    print(f"  test set: {len(test_idx)} queries\n")

    print("Scoring probes on test set...")
    tool_proba = tool_probe.predict_proba(X_te)
    tool_classes = tool_probe.classes_
    top1 = topk_labels(tool_proba, tool_classes, 1)[:, 0]
    top5 = topk_labels(tool_proba, tool_classes, 5)
    top10 = topk_labels(tool_proba, tool_classes, 10)

    agent_pred = orch_probe.predict(X_te)

    n = len(test_idx)
    recall5 = np.array([yt_te[i] in top5[i] for i in range(n)])
    recall10 = np.array([yt_te[i] in top10[i] for i in range(n)])
    top1_correct = top1 == yt_te
    agent_correct = agent_pred == ya_te
    # "probe had it but top-1 missed it": gt in top-10 yet top-1 is wrong.
    agent_failure = recall10 & ~top1_correct

    overall = metric_block(recall5, recall10, top1_correct, agent_correct,
                           agent_failure)

    print("\n==================== OVERALL ====================")
    print(f"Probe Top-1:        {overall['top1']:.4f}")
    print(f"Probe Recall@5:     {overall['recall5']:.4f}")
    print(f"Probe Recall@10:    {overall['recall10']:.4f}")
    print(f"Agent Accuracy:     {overall['agent_acc']:.4f}")
    print(f"Agent Failure Rate: {overall['agent_failure']:.4f} "
          f"(probe had it, top-1 missed)")

    print("\n================ PER-AGENT BREAKDOWN ================")
    agents = sorted(np.unique(ya_te))
    header = (f"{'agent':<18}{'n':>5}{'top1':>9}{'r@5':>9}"
              f"{'r@10':>9}{'agentAcc':>10}{'failRate':>10}")
    print(header)
    print("-" * len(header))
    for aid in agents:
        m = ya_te == aid
        b = metric_block(recall5[m], recall10[m], top1_correct[m],
                         agent_correct[m], agent_failure[m])
        print(f"{aid:<18}{b['n']:>5}{b['top1']:>9.4f}{b['recall5']:>9.4f}"
              f"{b['recall10']:>9.4f}{b['agent_acc']:>10.4f}"
              f"{b['agent_failure']:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
