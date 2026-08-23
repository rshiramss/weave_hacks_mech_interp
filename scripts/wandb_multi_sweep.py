"""Register one W&B *sweep per model* over the tool-routing probe hyperparameters.

This populates the project's "Sweeps" page with several distinct sweeps (one for
each base model whose hidden states we cached), each containing many runs across
a grid of C values and probe layers. All runs use cached hidden states only --
no model loading / GPU.

What you get on https://wandb.ai/<entity>/<project>/sweeps :
  * tool-probe-Csweep-Qwen2.5-7B
  * tool-probe-Csweep-Llama-3.1-8B
  * tool-probe-Csweep-Mistral-7B-v0.3
each with one run per (layer, C) combination, logging top-1 accuracy,
recall@5, and the agent-probe top-1 for context.

Run from the project root:
    python scripts/wandb_multi_sweep.py
"""

import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import wandb
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

QUERIES_PATH = "data/queries_mixed.json"
SEED = 42

# Each model -> {layer: cache_path}. Only layers present on disk are swept.
MODELS = {
    "Qwen2.5-7B": {
        24: "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy",
        27: "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer27.npy",
    },
    "Llama-3.1-8B": {
        24: "data/probe_cache/mixed_Llama-3.1-8B-Instruct_layer24.npy",
        27: "data/probe_cache/mixed_Llama-3.1-8B-Instruct_layer27.npy",
    },
    "Mistral-7B-v0.3": {
        24: "data/probe_cache/mixed_Mistral-7B-Instruct-v0.3_layer24.npy",
        27: "data/probe_cache/mixed_Mistral-7B-Instruct-v0.3_layer27.npy",
    },
}

C_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# ---- module globals populated per sweep / per run ----
WANDB_ENTITY = None
WANDB_PROJECT = None
CURRENT_MODEL = None          # model name the active sweep belongs to
_Y_TOOL = None                # tool labels (aligned to query order)
_Y_AGENT = None               # agent labels
_SPLIT_IDX = None             # (train_idx, test_idx) reused across runs
_CACHE_MEM = {}               # (model, layer) -> np.ndarray, loaded lazily


def _wandb_setup():
    global WANDB_ENTITY, WANDB_PROJECT
    load_dotenv()
    key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
    if not key:
        raise KeyError("Set WANDB_KEY (or WANDB_API_KEY) in your environment/.env")
    os.environ["WANDB_API_KEY"] = key
    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "abrahambhatti525-santa-clara-university")
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ToolOptim")
    wandb.login(key=key)


def _load_labels_and_split():
    global _Y_TOOL, _Y_AGENT, _SPLIT_IDX
    with open(QUERIES_PATH) as f:
        rows = json.load(f)
    _Y_TOOL = np.array([r["tool_id"] for r in rows])
    _Y_AGENT = np.array([r["agent_id"] for r in rows])
    idx = np.arange(len(rows))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=_Y_TOOL)
    _SPLIT_IDX = (tr, te)


def _get_cache(model, layer):
    key = (model, layer)
    if key not in _CACHE_MEM:
        path = MODELS[model][layer]
        X = np.load(path)
        if X.shape[0] != len(_Y_TOOL):
            raise ValueError(f"{path}: rows {X.shape[0]} != labels {len(_Y_TOOL)}")
        _CACHE_MEM[key] = X
    return _CACHE_MEM[key]


def recall_at_k(clf, X_test, y_test, k):
    proba = clf.predict_proba(X_test)
    top_k = np.argsort(-proba, axis=1)[:, :k]
    classes = clf.classes_
    hits = 0
    for i, true_label in enumerate(y_test):
        loc = np.where(classes == true_label)[0]
        if len(loc) and loc[0] in top_k[i]:
            hits += 1
    return hits / len(y_test)


def train_eval():
    run = wandb.init(group=f"sweep_{CURRENT_MODEL}", tags=["sweep", "tool-probe", CURRENT_MODEL])
    cfg = wandb.config
    layer = int(cfg.layer)
    C = float(cfg.C)
    run.name = f"{CURRENT_MODEL}-L{layer}-C{C}"
    run.config.update({"model": CURRENT_MODEL}, allow_val_change=True)

    X = _get_cache(CURRENT_MODEL, layer)
    tr, te = _SPLIT_IDX
    X_tr, X_te = X[tr], X[te]
    yt_tr, yt_te = _Y_TOOL[tr], _Y_TOOL[te]
    ya_tr, ya_te = _Y_AGENT[tr], _Y_AGENT[te]

    tool_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="lbfgs", max_iter=1000),
    )
    tool_clf.fit(X_tr, yt_tr)
    tool_top1 = accuracy_score(yt_te, tool_clf.predict(X_te))
    tool_r5 = recall_at_k(tool_clf, X_te, yt_te, 5)

    agent_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="lbfgs", max_iter=1000),
    )
    agent_clf.fit(X_tr, ya_tr)
    agent_top1 = accuracy_score(ya_te, agent_clf.predict(X_te))

    wandb.log({
        "tool_top1": tool_top1,
        "recall@5": tool_r5,
        "agent_top1": agent_top1,
        "C": C,
        "layer": layer,
    })
    run.finish()


def main():
    global CURRENT_MODEL
    _wandb_setup()
    _load_labels_and_split()

    sweep_urls = []
    for model, layer_paths in MODELS.items():
        layers = [L for L, p in layer_paths.items() if os.path.exists(p)]
        if not layers:
            print(f"[skip] {model}: no cached layers on disk")
            continue
        CURRENT_MODEL = model
        sweep_config = {
            "name": f"tool-probe-Csweep-{model}",
            "method": "grid",
            "metric": {"name": "recall@5", "goal": "maximize"},
            "parameters": {
                "C": {"values": C_VALUES},
                "layer": {"values": sorted(layers)},
            },
        }
        n_runs = len(C_VALUES) * len(layers)
        print(f"\n=== Sweep for {model}: {n_runs} runs "
              f"(C={C_VALUES} x layers={sorted(layers)}) ===")
        sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
        url = f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps/{sweep_id}"
        print(f"  sweep id: {sweep_id}\n  {url}")
        sweep_urls.append((model, url))
        wandb.agent(sweep_id, function=train_eval, count=n_runs)

    print("\nAll sweeps done. View them on the Sweeps page:")
    for model, url in sweep_urls:
        print(f"  {model:<18} {url}")
    print(f"\nSweeps page: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps")


if __name__ == "__main__":
    main()
