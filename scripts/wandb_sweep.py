"""W&B hyperparameter sweep for the tool routing probe (cached hidden states).

No model loading / GPU. Defines a grid sweep over LogisticRegression
hyperparameters and runs all combinations via wandb.agent, logging top-1
accuracy and recall@5 for each. Stays grouped with the other tuning runs.

Run from the project root:
    python scripts/wandb_sweep.py
"""

import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import wandb
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CACHE_PATH = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy"
QUERIES_PATH = "data/queries_mixed.json"
SEED = 42

# Filled in by load_split() before the agent starts.
X_tr = X_te = y_tr = y_te = None

sweep_config = {
    "method": "grid",
    "metric": {"name": "recall@5", "goal": "maximize"},
    "parameters": {
        "C": {"values": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]},
        "solver": {"values": ["lbfgs", "saga"]},
        "class_weight": {"values": ["balanced", None]},
    },
}


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
# Data
# ---------------------------------------------------------------------------

def recall_at_k(clf, X_test, y_test, k):
    proba = clf.predict_proba(X_test)
    top_k = np.argsort(-proba, axis=1)[:, :k]
    classes = clf.classes_
    hits = 0
    for i, true_label in enumerate(y_test):
        idx = np.where(classes == true_label)[0]
        if len(idx) and idx[0] in top_k[i]:
            hits += 1
    return hits / len(y_test)


def load_split():
    """Load cache + labels and create the 80/20 stratified split (seed=42)."""
    global X_tr, X_te, y_tr, y_te

    X = np.load(CACHE_PATH)
    with open(QUERIES_PATH) as f:
        rows = json.load(f)
    y = np.array([r["tool_id"] for r in rows])
    if X.shape[0] != len(rows):
        raise ValueError(
            f"cache rows ({X.shape[0]}) != query rows ({len(rows)})."
        )

    idx = np.arange(len(rows))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y
    )
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Sweep train function
# ---------------------------------------------------------------------------

def train_eval():
    run = wandb.init(group="probe_tuning", tags=["sweep", "logreg"])
    cfg = wandb.config

    clf = LogisticRegression(
        C=cfg.C,
        solver=cfg.solver,
        class_weight=cfg.class_weight,
        max_iter=2000,
    )
    clf.fit(X_tr, y_tr)

    top1 = accuracy_score(y_te, clf.predict(X_te))
    r5 = recall_at_k(clf, X_te, y_te, 5)
    wandb.log({"top1_acc": top1, "recall@5": r5})
    run.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    entity, project = _wandb_setup()
    load_split()

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    wandb.agent(sweep_id, function=train_eval, count=None)


if __name__ == "__main__":
    main()
