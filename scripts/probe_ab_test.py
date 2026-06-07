"""
A/B test of agent-probe vs tool-probe accuracy on two query datasets
(natural-language vs mixed phrasing), using cached hidden states at layer 27.

Runs over multiple models in sequence (see MODELS), loading and unloading each
from GPU memory between runs. Each model gets its own per-dataset cache file.
"""

import os
import gc
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import wandb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# W&B credentials (same load_dotenv() / WANDB_API_KEY pattern as the other
# scripts). WANDB_KEY is read first and mirrored into WANDB_API_KEY; the entity
# and project come from .env too, with sensible defaults.
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
# Hardcoded paths
# ---------------------------------------------------------------------------

REGISTRY_PATH = "data/registry.json"
QUERIES_NL_PATH = "data/queries_nl.json"
QUERIES_MIXED_PATH = "data/queries_mixed.json"
CACHE_DIR = "data/probe_cache"

DATASETS = {
    "nl": QUERIES_NL_PATH,
    "mixed": QUERIES_MIXED_PATH,
}

# ---------------------------------------------------------------------------
# Model / extraction config
# ---------------------------------------------------------------------------

# Models to evaluate in sequence. Each is loaded, run over all datasets, then
# unloaded from GPU memory before the next one. All use the same LAYER for now.
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# Set per-model inside main(); defaults to the first model for standalone use.
MODEL_NAME = MODELS[0]
LAYER = 27
BATCH_SIZE = 16
MAX_LENGTH = 128

# Sweep every 4th layer — balances coverage vs runtime
LAYER_STEP = 4


def model_shortname(model_name: str = None) -> str:
    name = model_name if model_name is not None else MODEL_NAME
    return name.split("/")[-1]

# ---------------------------------------------------------------------------
# Probe config
# ---------------------------------------------------------------------------

PROBE_KW = dict(C=0.01, solver="lbfgs", max_iter=2000, class_weight="balanced")


def require_file(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {what} at '{path}'. "
            f"Generate this file before running probe_ab_test.py."
        )


def load_registry():
    require_file(REGISTRY_PATH, "registry")
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    agent_ids = sorted({a["agent_id"] for a in registry["agents"]})
    tool_ids = sorted({t["tool_id"] for t in registry["tools"]})
    tool_to_agent = {t["tool_id"]: t["agent_id"] for t in registry["tools"]}
    return agent_ids, tool_ids, tool_to_agent


def load_queries(path: str, tool_to_agent: dict):
    require_file(path, "query dataset")
    with open(path) as f:
        raw = json.load(f)
    queries = [r["query_text"] for r in raw]
    tools = [r["tool_id"] for r in raw]
    agents = [tool_to_agent[t] for t in tools]
    return queries, tools, agents


# ---------------------------------------------------------------------------
# Hidden state extraction (cached)
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        output_hidden_states=True,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    if device == "cpu":
        model = model.to(device)

    _tokenizer, _model = tokenizer, model
    return tokenizer, model


def unload_model():
    """Free the currently loaded model from (GPU) memory before loading the next."""
    global _tokenizer, _model
    if _model is None:
        return

    import torch

    del _model
    del _tokenizer
    _model = None
    _tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_hidden_states(queries: list[str]) -> np.ndarray:
    import torch

    tokenizer, model = _get_model()
    device = next(model.parameters()).device

    vectors = []
    for start in range(0, len(queries), BATCH_SIZE):
        batch = queries[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # (num_layers + 1) tensors of (B, seq, H)
        hs = hidden_states[LAYER + 1]  # output of transformer block LAYER (0-indexed)

        attention_mask = inputs["attention_mask"]
        # left padding => last real token is always the final position
        last_tok = hs[:, -1, :]
        vectors.extend(last_tok.float().cpu().numpy())

        done = min(start + BATCH_SIZE, len(queries))
        print(f"  Extracted {done}/{len(queries)}", end="\r")
    print()
    return np.stack(vectors)


def get_or_extract_hidden_states(name: str, queries: list[str]) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        CACHE_DIR, f"{name}_{model_shortname()}_layer{LAYER}.npy"
    )
    if os.path.exists(cache_path):
        X = np.load(cache_path)
        if X.shape[0] != len(queries):
            print(f"  Cache shape mismatch for '{name}': {X.shape[0]} rows cached, {len(queries)} queries. Reextracting...")
            os.remove(cache_path)
        else:
            print(f"Loading cached hidden states for '{name}' from {cache_path}")
            return X

    print(f"Extracting hidden states for '{name}' ({len(queries)} queries) ...")
    X = extract_hidden_states(queries)
    np.save(cache_path, X)
    print(f"Cached hidden states to {cache_path}")
    return X


# ---------------------------------------------------------------------------
# Probe training / evaluation
# ---------------------------------------------------------------------------

def recall_at_k(clf, X_te, y_te, k):
    proba = clf.predict_proba(X_te)
    top_k = np.argsort(-proba, axis=1)[:, :k]
    classes = clf.classes_
    hits = 0
    for i, true_label in enumerate(y_te):
        true_class_idx = np.where(classes == true_label)[0]
        if len(true_class_idx) and true_class_idx[0] in top_k[i]:
            hits += 1
    return hits / len(y_te)


def train_eval_probe(X_tr, X_te, y_tr, y_te, recall_k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    clf = LogisticRegression(**PROBE_KW)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    return {
        "top1_acc": accuracy_score(y_te, y_pred),
        "macro_f1": f1_score(y_te, y_pred, average="macro"),
        f"recall@{recall_k}": recall_at_k(clf, X_te, y_te, recall_k),
    }


def stratified_split_indices(tools: np.ndarray, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(tools))
    # filter out tools with only 1 sample
    counts = {}
    for t in tools:
        counts[t] = counts.get(t, 0) + 1
    mask = np.array([counts[t] >= 2 for t in tools])
    
    filtered_idx = idx[mask]
    filtered_tools = tools[mask]
    
    dropped = idx[~mask]
    if len(dropped) > 0:
        print(f"  Warning: dropped {len(dropped)} samples with singleton tool labels")

    train_idx, test_idx = train_test_split(
        filtered_idx, test_size=test_size, random_state=seed, stratify=filtered_tools
    )
    return train_idx, test_idx


def run_dataset(name: str, path: str, tool_to_agent: dict):
    print(f"\n=== Dataset: {name} ({path}) ===")
    queries, tools, agents = load_queries(path, tool_to_agent)
    queries = np.array(queries)
    tools = np.array(tools)
    agents = np.array(agents)

    X = get_or_extract_hidden_states(name, list(queries))

    train_idx, test_idx = stratified_split_indices(tools)
    X_tr, X_te = X[train_idx], X[test_idx]

    agent_results = train_eval_probe(
        X_tr, X_te, agents[train_idx], agents[test_idx], recall_k=2
    )
    tool_results = train_eval_probe(
        X_tr, X_te, tools[train_idx], tools[test_idx], recall_k=5
    )
    return agent_results, tool_results


def print_comparison_table(all_results: dict):
    print(f"\n=== Probe A/B comparison (layer {LAYER}) ===")
    header = (
        f"{'Model':<26} {'Dataset':<8} {'Probe':<8} "
        f"{'Top-1 Acc':>10} {'Macro F1':>10} {'Recall@K':>10}"
    )
    print(header)
    print("-" * len(header))
    for model_name, results in all_results.items():
        short = model_shortname(model_name)
        for dataset_name, (agent_res, tool_res) in results.items():
            agent_recall_key = [k for k in agent_res if k.startswith("recall@")][0]
            tool_recall_key = [k for k in tool_res if k.startswith("recall@")][0]
            print(
                f"{short:<26} {dataset_name:<8} {'agent':<8} "
                f"{agent_res['top1_acc']:>10.4f} {agent_res['macro_f1']:>10.4f} "
                f"{agent_res[agent_recall_key]:>10.4f} ({agent_recall_key})"
            )
            print(
                f"{short:<26} {dataset_name:<8} {'tool':<8} "
                f"{tool_res['top1_acc']:>10.4f} {tool_res['macro_f1']:>10.4f} "
                f"{tool_res[tool_recall_key]:>10.4f} ({tool_recall_key})"
            )


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def log_ab_results_to_wandb(all_results: dict):
    """Log the A/B comparison table to W&B as a single run.

    One wandb.Table with every (Model, Dataset, Probe) row plus a bar chart of
    Top-1 accuracy grouped by Model+Dataset.
    """
    entity, project = _wandb_setup()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run = wandb.init(
        entity=entity,
        project=project,
        name=f"probe-ab-{ts}",
        group="probe_ab",
        tags=["probe_ab", "ab_comparison"],
        config={"layer": LAYER, "models": MODELS},
        reinit=True,
    )

    columns = ["Model", "Dataset", "Probe", "Top1-Acc", "Macro-F1", "RecallK"]
    table = wandb.Table(columns=columns)
    bar_rows = []  # [label, top1] grouped by Model+Dataset(+Probe)

    for model_name, results in all_results.items():
        short = model_shortname(model_name)
        for dataset_name, (agent_res, tool_res) in results.items():
            for probe, res in (("agent", agent_res), ("tool", tool_res)):
                recall_key = [k for k in res if k.startswith("recall@")][0]
                table.add_data(
                    short,
                    dataset_name,
                    probe,
                    res["top1_acc"],
                    res["macro_f1"],
                    res[recall_key],
                )
                bar_rows.append(
                    [f"{short}/{dataset_name}/{probe}", res["top1_acc"]]
                )

    bar_table = wandb.Table(data=bar_rows, columns=["Model+Dataset", "Top1-Acc"])
    wandb.log({
        "ab_comparison_table": table,
        "top1_by_model_dataset": wandb.plot.bar(
            bar_table, "Model+Dataset", "Top1-Acc",
            title="Top-1 Accuracy by Model+Dataset",
        ),
    })
    wandb.finish()
    print(f"Logged W&B run 'probe-ab-{ts}'.")


def log_layer_sweep_to_wandb(sweep_summary: dict, sweep_curves: dict):
    """Log the layer-sweep results to W&B as a single run.

    Includes the best-layer summary table, a Tool-Recall5-by-Model bar chart,
    and a per-model line chart of tool/agent recall across swept layers.
    """
    entity, project = _wandb_setup()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run = wandb.init(
        entity=entity,
        project=project,
        name=f"layer-sweep-{ts}",
        group="layer_sweep",
        tags=["layer_sweep", "summary"],
        config={"dataset": "mixed", "layer_step": LAYER_STEP, "models": MODELS},
        reinit=True,
    )

    columns = ["Model", "Best-Layer", "Tool-Recall5", "Agent-Recall2"]
    table = wandb.Table(columns=columns)
    bar_rows = []
    for model_name, (layer, tool_recall, agent_recall) in sweep_summary.items():
        short = model_shortname(model_name)
        table.add_data(short, layer, tool_recall, agent_recall)
        bar_rows.append([short, tool_recall])

    bar_table = wandb.Table(data=bar_rows, columns=["Model", "Tool-Recall5"])
    log_payload = {
        "layer_sweep_summary_table": table,
        "tool_recall5_by_model": wandb.plot.bar(
            bar_table, "Model", "Tool-Recall5", title="Tool Recall@5 by Model"
        ),
    }

    # Per-model line chart: x = layer, y = tool_recall5 and agent_recall2.
    for model_name, points in sweep_curves.items():
        short = model_shortname(model_name)
        layers = [p[0] for p in points]
        tool_series = [p[1] for p in points]
        agent_series = [p[2] for p in points]
        log_payload[f"{short}/layer_sweep_curve"] = wandb.plot.line_series(
            xs=layers,
            ys=[tool_series, agent_series],
            keys=["tool_recall@5", "agent_recall@2"],
            title=f"{short}: recall vs layer",
            xname="layer",
        )

    wandb.log(log_payload)
    wandb.finish()
    print(f"Logged W&B run 'layer-sweep-{ts}'.")


# ---------------------------------------------------------------------------
# Layer sweep (mixed dataset only)
# ---------------------------------------------------------------------------

def run_layer_sweep():
    """Sweep transformer layers for each model on the mixed dataset and report
    the best layer (by tool probe recall@5). Additive to the main A/B test."""
    global MODEL_NAME, LAYER

    _, _, tool_to_agent = load_registry()

    print("\n########## LAYER SWEEP (mixed dataset only) ##########")

    queries, tools, agents = load_queries(QUERIES_MIXED_PATH, tool_to_agent)
    queries = np.array(queries)
    tools = np.array(tools)
    agents = np.array(agents)

    train_idx, test_idx = stratified_split_indices(tools)

    # model_name -> (best_layer, best_tool_recall@5, agent_recall@2 at best_layer)
    sweep_summary = {}
    # model_name -> [(layer, tool_recall@5, agent_recall@2), ...] per swept layer
    sweep_curves = {}

    for model_name in MODELS:
        MODEL_NAME = model_name

        # Load the model first so we can read its actual architecture.
        _get_model()
        num_layers = _model.config.num_hidden_layers
        layers_to_sweep = list(range(0, num_layers, LAYER_STEP))
        print(f"\n{model_name}: {num_layers} layers total, sweeping {layers_to_sweep}")

        best = None  # (layer, tool_recall, agent_recall)
        curve = []  # (layer, tool_recall, agent_recall) per layer
        for layer in layers_to_sweep:
            LAYER = layer
            X = get_or_extract_hidden_states("mixed", list(queries))
            X_tr, X_te = X[train_idx], X[test_idx]

            agent_res = train_eval_probe(
                X_tr, X_te, agents[train_idx], agents[test_idx], recall_k=2
            )
            tool_res = train_eval_probe(
                X_tr, X_te, tools[train_idx], tools[test_idx], recall_k=5
            )

            tool_recall = tool_res["recall@5"]
            agent_recall = agent_res["recall@2"]
            print(
                f"  layer {layer:>3}: tool recall@5 = {tool_recall:.4f}, "
                f"agent recall@2 = {agent_recall:.4f}"
            )
            curve.append((layer, tool_recall, agent_recall))

            if best is None or tool_recall > best[1]:
                best = (layer, tool_recall, agent_recall)

        print(
            f"  >>> best layer for {model_shortname(model_name)}: "
            f"layer {best[0]} (tool recall@5 = {best[1]:.4f})"
        )
        sweep_summary[model_name] = best
        sweep_curves[model_name] = curve

        unload_model()

    print("\n=== Layer sweep summary (mixed dataset) ===")
    header = (
        f"{'Model':<30} {'Best Layer':<12} "
        f"{'Tool Recall@5':<15} {'Agent Recall@2':<15}"
    )
    print(header)
    print("-" * len(header))
    for model_name, (layer, tool_recall, agent_recall) in sweep_summary.items():
        print(
            f"{model_shortname(model_name):<30} {('layer ' + str(layer)):<12} "
            f"{tool_recall:<15.3f} {agent_recall:<15.3f}"
        )

    try:
        log_layer_sweep_to_wandb(sweep_summary, sweep_curves)
    except Exception as e:
        print(f"  [warn] W&B logging for layer sweep failed: {e}")


def main():
    global MODEL_NAME, LAYER

    agent_ids, tool_ids, tool_to_agent = load_registry()
    print(f"Registry loaded: {len(agent_ids)} agents, {len(tool_ids)} tools")

    all_results = {}
    for model_name in MODELS:
        MODEL_NAME = model_name
        LAYER = 27
        print(f"\n########## Model: {MODEL_NAME} (layer {LAYER}) ##########")

        results = {}
        for name, path in DATASETS.items():
            results[name] = run_dataset(name, path, tool_to_agent)
        all_results[model_name] = results

        unload_model()

    print_comparison_table(all_results)

    try:
        log_ab_results_to_wandb(all_results)
    except Exception as e:
        print(f"  [warn] W&B logging for A/B comparison failed: {e}")


if __name__ == "__main__":
    main()
    run_layer_sweep()
