"""Validate the saved routing probe(s) on genuinely held-out data.

Reproduces the EXACT train_test_split used in train_best_probes.py (same arrays,
same seed, same stratify) and evaluates only on the 20% test split — rows the
probe never saw during training, but drawn from the same distribution ("similar
but not memorized"). Uses cached hidden states only: no GPU, no model loading.

Run from the project root:
    python scripts/validate_probe.py
"""

import os
import sys
import json

import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CACHE_PATH = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy"
QUERIES_PATH = "data/queries_mixed.json"
TOOL_PROBE_PATH = "data/probes/tool_probe_best.pkl"
ORCH_PROBE_PATH = "data/probes/orchestrator_probe_best.pkl"
SEED = 42


def recall_at_k(clf, X_te, y_te, k):
    proba = clf.predict_proba(X_te)
    top_k = np.argsort(-proba, axis=1)[:, :k]
    classes = clf.classes_
    hits = 0
    for i, true_label in enumerate(y_te):
        idx = np.where(classes == true_label)[0]
        if len(idx) and idx[0] in top_k[i]:
            hits += 1
    return hits / len(y_te)


def main():
    X = np.load(CACHE_PATH)
    with open(QUERIES_PATH) as f:
        rows = json.load(f)
    y_tool = np.array([r["tool_id"] for r in rows])
    y_agent = np.array([r["agent_id"] for r in rows])

    if X.shape[0] != len(rows):
        print(f"ERROR: cache rows ({X.shape[0]}) != query rows ({len(rows)}).")
        sys.exit(1)

    # Reproduce the exact split from train_best_probes.py so X_te / y*_te are the
    # held-out 20% the probes never trained on.
    X_tr, X_te, yt_tr, yt_te, ya_tr, ya_te = train_test_split(
        X, y_tool, y_agent, test_size=0.2, random_state=SEED, stratify=y_tool
    )
    print(
        f"Held-out test set: {len(X_te)} queries "
        f"(20% of {len(rows)}, never seen in training)."
    )

    if not os.path.exists(TOOL_PROBE_PATH):
        print(f"ERROR: tool probe not found at '{TOOL_PROBE_PATH}'.")
        sys.exit(1)

    tool = joblib.load(TOOL_PROBE_PATH)
    tool_top1 = accuracy_score(yt_te, tool.predict(X_te))
    tool_r5 = recall_at_k(tool, X_te, yt_te, 5)
    print("\n=== Tool probe (200-class) ===")
    print(f"  Top-1 accuracy: {tool_top1:.4f}")
    print(f"  Recall@5:       {tool_r5:.4f}")

    if os.path.exists(ORCH_PROBE_PATH):
        orch = joblib.load(ORCH_PROBE_PATH)
        orch_top1 = accuracy_score(ya_te, orch.predict(X_te))
        orch_r2 = recall_at_k(orch, X_te, ya_te, 2)
        print("\n=== Orchestrator probe (agent) ===")
        print(f"  Top-1 accuracy: {orch_top1:.4f}")
        print(f"  Recall@2:       {orch_r2:.4f}")
    else:
        print(f"\n(Orchestrator probe not found at '{ORCH_PROBE_PATH}'; skipped.)")


if __name__ == "__main__":
    main()
