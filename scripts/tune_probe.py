"""Hyperparameter sweep for the tool routing probe on cached hidden states.

No model loading / GPU. Sweeps LogReg C, prints a ranked table by recall@5, and
saves the best probe.

Run from the project root:
    python scripts/tune_probe.py
"""

import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import wandb
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CACHE_PATH = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy"
QUERIES_PATH = "data/queries_mixed.json"
BEST_PATH = "data/probes/tool_probe_best.pkl"
SEED = 42

C_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


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


def evaluate(clf, X_tr, X_te, y_tr, y_te):
    clf.fit(X_tr, y_tr)
    top1 = accuracy_score(y_te, clf.predict(X_te))
    r5 = recall_at_k(clf, X_te, y_te, 5)
    return clf, top1, r5


# ---------------------------------------------------------------------------
# W&B logging (same load_dotenv() / WANDB_API_KEY + wandb.init() pattern as
# probe_ab_test.py's _wandb_setup() / log_*_to_wandb()).
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


def log_tuning_to_wandb(logreg_results):
    """Log the hyperparameter sweep results to W&B as a single run.

    logreg_results: list of (C, top1, recall5) from the C sweep.
    """
    entity, project = _wandb_setup()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run = wandb.init(
        entity=entity,
        project=project,
        name=f"probe-tuning-{ts}",
        group="probe_tuning",
        tags=["probe_tuning", "hyperparameter_search"],
        config={"C_values": C_VALUES},
        reinit=True,
    )

    columns = ["Type", "Config", "Top1-Acc", "Recall@5"]
    table = wandb.Table(columns=columns)
    bar_rows = []
    for c, top1, r5 in logreg_results:
        cfg = f"C={c}"
        table.add_data("LogReg", cfg, top1, r5)
        bar_rows.append([cfg, r5])

    bar_table = wandb.Table(data=bar_rows, columns=["Config", "Recall@5"])
    curve_table = wandb.Table(
        data=[[float(c), r5] for c, _, r5 in logreg_results],
        columns=["C", "Recall@5"],
    )
    wandb.log({
        "tuning_table": table,
        "recall5_by_config": wandb.plot.bar(
            bar_table, "Config", "Recall@5", title="Recall@5 by Config"
        ),
        "logreg_recall5_vs_C": wandb.plot.line(
            curve_table, "C", "Recall@5", title="LogReg recall@5 vs C"
        ),
    })

    best_logreg_recall5 = max((r5 for _, _, r5 in logreg_results), default=None)
    run.summary.update({"best_logreg_recall5": best_logreg_recall5})
    wandb.finish()
    print(f"Logged W&B run 'probe-tuning-{ts}'.")


def main():
    X = np.load(CACHE_PATH)
    with open(QUERIES_PATH) as f:
        rows = json.load(f)
    y = np.array([r["tool_id"] for r in rows])

    if X.shape[0] != len(rows):
        print(f"ERROR: cache rows ({X.shape[0]}) != query rows ({len(rows)}).")
        sys.exit(1)

    idx = np.arange(len(rows))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y
    )
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    results = []  # (name, top1, recall5, model)
    logreg_results = []  # (C, top1, recall5) for W&B logging

    # --- Sweep 1: LogReg C ---------------------------------------------------
    print("=== Sweep 1: LogReg C ===")
    best_c, best_c_r5 = C_VALUES[0], -1.0
    for c in C_VALUES:
        clf = LogisticRegression(
            C=c, solver="lbfgs", max_iter=2000, class_weight="balanced"
        )
        clf, top1, r5 = evaluate(clf, X_tr, X_te, y_tr, y_te)
        print(f"  C={c:<7} top-1={top1:.4f}  recall@5={r5:.4f}")
        results.append((f"LogReg C={c} (lbfgs)", top1, r5, clf))
        logreg_results.append((c, top1, r5))
        if r5 > best_c_r5:
            best_c, best_c_r5 = c, r5
    print(f"  -> best C = {best_c} (recall@5={best_c_r5:.4f})")

    # --- Ranked table --------------------------------------------------------
    results.sort(key=lambda r: r[2], reverse=True)
    print("\n=== All results ranked by recall@5 ===")
    header = f"{'Config':<28}{'Top-1 Acc':>11}{'Recall@5':>11}"
    print(header)
    print("-" * len(header))
    for name, top1, r5, _ in results:
        print(f"{name:<28}{top1:>11.4f}{r5:>11.4f}")

    best_name, best_top1, best_r5, best_model = results[0]
    os.makedirs(os.path.dirname(BEST_PATH), exist_ok=True)
    joblib.dump(best_model, BEST_PATH)
    print(
        f"\nBest: {best_name} (recall@5={best_r5:.4f}, top-1={best_top1:.4f})"
        f"\nSaved to {BEST_PATH}"
    )

    try:
        log_tuning_to_wandb(logreg_results)
    except Exception as e:
        print(f"  [warn] W&B logging for probe tuning failed: {e}")


if __name__ == "__main__":
    main()
