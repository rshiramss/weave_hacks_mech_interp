"""Train and save the orchestrator (agent) and tool routing probes.

Uses the existing cached layer-24 hidden states (no model loading / GPU). Saves
both LogisticRegression probes to data/probes/ for FastAPI integration.

Run from the project root:
    python scripts/train_probes.py
"""

import json
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CACHE_PATH = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy"
QUERIES_PATH = "data/queries_mixed.json"
PROBES_DIR = "data/probes"
SEED = 42

PROBE_KW = dict(C=0.01, solver="lbfgs", max_iter=2000, class_weight="balanced")


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

    idx = np.arange(len(rows))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y_tool
    )
    X_tr, X_te = X[train_idx], X[test_idx]

    orchestrator_probe = LogisticRegression(**PROBE_KW)
    orchestrator_probe.fit(X_tr, y_agent[train_idx])
    tool_probe = LogisticRegression(**PROBE_KW)
    tool_probe.fit(X_tr, y_tool[train_idx])

    orch_acc = accuracy_score(y_agent[test_idx], orchestrator_probe.predict(X_te))
    orch_r2 = recall_at_k(orchestrator_probe, X_te, y_agent[test_idx], 2)
    tool_acc = accuracy_score(y_tool[test_idx], tool_probe.predict(X_te))
    tool_r5 = recall_at_k(tool_probe, X_te, y_tool[test_idx], 5)

    print(f"Orchestrator: top-1 acc = {orch_acc:.4f}, recall@2 = {orch_r2:.4f}")
    print(f"Tool:         top-1 acc = {tool_acc:.4f}, recall@5 = {tool_r5:.4f}")

    os.makedirs(PROBES_DIR, exist_ok=True)
    joblib.dump(orchestrator_probe, os.path.join(PROBES_DIR, "orchestrator_probe.pkl"))
    joblib.dump(tool_probe, os.path.join(PROBES_DIR, "tool_probe.pkl"))

    print("Probes saved. Ready for FastAPI integration.")


if __name__ == "__main__":
    main()
