"""
One-shot results logger.

Reads raw benchmark terminal output pasted into `results.txt` in the project
root, parses the two summary tables, and logs each as a separate W&B run under
the "ToolOptim" project:

  1. Router sweep summary (from benchmark_llama_tool_selection.py)
       -> run "router-sweep-results"
  2. Layer sweep summary (from probe_ab_test.py)
       -> run "layer-sweep-results"

Each run gets per-row summary metrics, a wandb.Table with the full results, and
a bar chart (E2E-Acc for the router sweep, Tool Recall@5 for the layer sweep).

Credentials follow the same load_dotenv() / WANDB_API_KEY pattern as the other
scripts (wandb_default.py, log_sweep_results.py). WANDB_KEY is read first and
mirrored into WANDB_API_KEY; WANDB_ENTITY / WANDB_PROJECT come from .env too.
"""

import os
import re

import wandb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results.txt")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Cells rendered as an em/en dash (or bare hyphen) are "not applicable" markers.
PLACEHOLDERS = {"\u2014", "\u2013", "\u2212", "-", "--"}

ROUTER_COLUMNS = [
    "Router", "Layer", "C", "TrainTop1", "TestTop1", "TestR@5", "E2E-Acc", "ms/q",
]
LAYER_COLUMNS = ["Model", "Best Layer", "Tool Recall@5", "Agent Recall@2"]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def load_wandb_config():
    load_dotenv(ENV_PATH)
    key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
    if not key:
        raise KeyError("Set WANDB_KEY (or WANDB_API_KEY) in your environment/.env")
    # weave/wandb both look for WANDB_API_KEY specifically.
    os.environ["WANDB_API_KEY"] = key

    entity = os.environ.get("WANDB_ENTITY", "abrahambhatti525-santa-clara-university")
    project = os.environ.get("WANDB_PROJECT", "ToolOptim")
    return key, entity, project


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _is_separator(line: str) -> bool:
    """True for ruler lines made only of dashes / box-drawing / block chars."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("\u2500\u2501-\u2014\u2013\u2588= ")


def _is_block_border(line: str) -> bool:
    return "\u2588" in line


def _is_prompt(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("(") and "@" in s:  # e.g. "(venv) (base) user@host:~/path"
        return True
    return s.endswith("$") and len(s) <= 2


def _parse_cell(token: str):
    if token in PLACEHOLDERS:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def parse_router_sweep(text: str) -> list[dict]:
    """Parse the 'ROUTER SWEEP SUMMARY' table.

    Columns: Router, Layer, C, TrainTop1, TestTop1, TestR@5, E2E-Acc, ms/q.
    The Router name may contain spaces (e.g. 'Naive (20 tools)'), so we treat
    the last 7 whitespace tokens as the numeric columns and join the rest as
    the router label.
    """
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Router" in line and "Layer" in line and "E2E-Acc" in line:
            header_idx = i
            break
    if header_idx is None:
        return []

    rows = []
    n_metric = len(ROUTER_COLUMNS) - 1  # everything except Router
    for line in lines[header_idx + 1:]:
        if _is_block_border(line):
            break
        if not line.strip() or _is_separator(line):
            continue
        tokens = line.split()
        if len(tokens) < len(ROUTER_COLUMNS):
            continue
        metric_tokens = tokens[-n_metric:]
        router = " ".join(tokens[:-n_metric])
        values = [_parse_cell(t) for t in metric_tokens]
        row = {"Router": router}
        row.update(dict(zip(ROUTER_COLUMNS[1:], values)))
        rows.append(row)
    return rows


def parse_layer_sweep(text: str) -> list[dict]:
    """Parse the 'Layer sweep summary' table.

    Columns: Model, Best Layer, Tool Recall@5, Agent Recall@2. Tolerant of
    terminal line-wrapping (e.g. the trailing Agent Recall@2 value spilling
    onto the next line) by joining the table region and regex-extracting rows.
    The Agent Recall@2 value is optional in case the capture was truncated.
    """
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Best Layer" in line and "Tool Recall" in line:
            header_idx = i
            break
    if header_idx is None:
        return []

    collected = []
    for line in lines[header_idx + 1:]:
        if not line.strip() or _is_separator(line):
            continue
        if _is_prompt(line) or line.lstrip().startswith("==="):
            break
        if _is_block_border(line) or "######" in line or line.strip().startswith("Done"):
            break
        collected.append(line.strip())

    joined = " ".join(collected)
    pattern = re.compile(r"(\S+)\s+layer\s+(\d+)\s+([\d.]+)(?:\s+([\d.]+))?")
    rows = []
    for m in pattern.finditer(joined):
        model, layer, tool_recall, agent_recall = m.groups()
        rows.append({
            "Model": model,
            "Best Layer": int(layer),
            "Tool Recall@5": float(tool_recall),
            "Agent Recall@2": float(agent_recall) if agent_recall is not None else None,
        })
    return rows


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def log_router_sweep(rows: list[dict], entity: str, project: str):
    run = wandb.init(
        entity=entity,
        project=project,
        name="router-sweep-results",
        group="router_sweep",
        tags=["router_sweep", "summary"],
        reinit=True,
    )

    table = wandb.Table(columns=ROUTER_COLUMNS)
    for r in rows:
        table.add_data(*[r.get(c) for c in ROUTER_COLUMNS])

    # Each row -> one summary metric dict, namespaced by router.
    for r in rows:
        key = _sanitize(r["Router"])
        metrics = {
            f"{key}/{col}": r[col]
            for col in ROUTER_COLUMNS[1:]
            if r.get(col) is not None
        }
        run.summary.update(metrics)

    bar = wandb.Table(
        data=[[r["Router"], r["E2E-Acc"]] for r in rows if r.get("E2E-Acc") is not None],
        columns=["Router", "E2E-Acc"],
    )
    wandb.log({
        "router_sweep_table": table,
        "e2e_acc_by_router": wandb.plot.bar(
            bar, "Router", "E2E-Acc", title="E2E Accuracy by Router"
        ),
    })
    wandb.finish()
    print(f"Logged 'router-sweep-results' with {len(rows)} rows.")


def log_layer_sweep(rows: list[dict], entity: str, project: str):
    run = wandb.init(
        entity=entity,
        project=project,
        name="layer-sweep-results",
        group="layer_sweep",
        tags=["layer_sweep", "summary"],
        reinit=True,
    )

    table = wandb.Table(columns=LAYER_COLUMNS)
    for r in rows:
        table.add_data(*[r.get(c) for c in LAYER_COLUMNS])

    for r in rows:
        key = _sanitize(r["Model"])
        metrics = {
            f"{key}/{col}": r[col]
            for col in LAYER_COLUMNS[1:]
            if r.get(col) is not None
        }
        run.summary.update(metrics)

    bar = wandb.Table(
        data=[[r["Model"], r["Tool Recall@5"]] for r in rows if r.get("Tool Recall@5") is not None],
        columns=["Model", "Tool Recall@5"],
    )
    wandb.log({
        "layer_sweep_table": table,
        "tool_recall5_by_model": wandb.plot.bar(
            bar, "Model", "Tool Recall@5", title="Tool Recall@5 by Model"
        ),
    })
    wandb.finish()
    print(f"Logged 'layer-sweep-results' with {len(rows)} rows.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Could not find results file at '{RESULTS_PATH}'. "
            f"Paste raw benchmark terminal output into results.txt first."
        )
    with open(RESULTS_PATH, encoding="utf-8", errors="replace") as f:
        text = f.read()

    key, entity, project = load_wandb_config()
    wandb.login(key=key)

    router_rows = parse_router_sweep(text)
    layer_rows = parse_layer_sweep(text)

    print(f"Parsed {len(router_rows)} router-sweep rows, {len(layer_rows)} layer-sweep rows.")

    if router_rows:
        log_router_sweep(router_rows, entity, project)
    else:
        print("No router sweep table found in results.txt; skipping that run.")

    if layer_rows:
        log_layer_sweep(layer_rows, entity, project)
    else:
        print("No layer sweep table found in results.txt; skipping that run.")

    print(f"Done. Check project '{project}' on wandb.ai.")


if __name__ == "__main__":
    main()
