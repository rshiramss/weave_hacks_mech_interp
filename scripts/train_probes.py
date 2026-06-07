"""Train the two linear probes from extracted activations (Stage 3).

Reads `activations.npy` ([N, 3584] float32) and an index-aligned `labels.jsonl`
(produced by `modal_app/extract_activations.py::batch`), fits two multinomial
LogisticRegression probes, reports the §6 gate metrics, and saves the artifacts.

Two probes, one activation matrix (mi_implementation.md §6):
  - agent probe  (orchestrator_probe.pkl) — 5 agent_id classes,  metric recall@2
  - tool  probe  (tool_probe.pkl)         — 200 tool_id classes, metric recall@5

Tool recall@5 is measured WITH masking (zero tools outside the gold agent before
top-5) so the number matches the production inference path (src/probes/inference.py)
— an effective 40-way pool per agent.

PARITY: probes are bare LogisticRegression fit on RAW h (no scaler) — inference
calls predict_proba on raw h directly. Do not add a StandardScaler here.
Requires scikit-learn >= 1.9.0 (older minors crash in predict_proba on these pkls).

    python scripts/train_probes.py \
        --activations data/activations_layer24.npy \
        --labels data/activation_labels.jsonl
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.probes.extract_h import (  # noqa: E402
    D_MODEL,
    LAYER,
    MAX_LENGTH,
    MODEL_ID,
)

REGISTRY_PATH = ROOT / "data" / "registry.json"

# §6 ship gate.
AGENT_RECALL_AT_2_GATE = 0.80
TOOL_RECALL_AT_5_GATE = 0.50

AGENT_PROBE_FILE = "orchestrator_probe.pkl"
TOOL_PROBE_FILE = "tool_probe.pkl"
CONFIG_FILE = "probe_config.json"

# Deterministic fallback split when labels carry no `split` tag.
DEFAULT_SEED = 1337
DEFAULT_VAL_FRAC = 0.10
DEFAULT_TEST_FRAC = 0.10


def _load_labels(path: Path) -> list[dict]:
    text = path.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _tool_to_agent() -> dict[str, str]:
    reg = json.loads(REGISTRY_PATH.read_text())
    return {t["tool_id"]: t["agent_id"] for t in reg["tools"]}


def _split_indices(rows: list[dict], seed: int, val_frac: float,
                   test_frac: float) -> dict[str, list[int]]:
    """Index lists per split. Uses the `split` tag if present, else stratifies
    a deterministic split by tool_id (mirrors the generator)."""
    if any(row.get("split") for row in rows):
        by_split: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            by_split[row.get("split", "train")].append(i)
        return by_split

    by_tool: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_tool[row["tool_id"]].append(i)

    rng = random.Random(seed)
    out: dict[str, list[int]] = defaultdict(list)
    for indices in by_tool.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        for rank, idx in enumerate(shuffled):
            if rank < n_test:
                out["test"].append(idx)
            elif rank < n_test + n_val:
                out["val"].append(idx)
            else:
                out["train"].append(idx)
    return out


def _recall_at_k(proba: np.ndarray, classes: np.ndarray,
                 gold: list[str], k: int) -> float:
    topk = np.argsort(-proba, axis=1)[:, :k]
    hits = sum(gold[i] in classes[topk[i]] for i in range(len(gold)))
    return hits / len(gold)


def _masked_tool_recall_at_5(proba: np.ndarray, classes: np.ndarray,
                             gold_tool: list[str], gold_agent: list[str],
                             tool_to_agent: dict[str, str]) -> float:
    owner = np.array([tool_to_agent.get(str(c)) for c in classes])
    hits = 0
    for i in range(len(gold_tool)):
        scores = np.where(owner == gold_agent[i], proba[i], 0.0)
        top5 = classes[np.argsort(-scores)[:5]]
        hits += gold_tool[i] in top5
    return hits / len(gold_tool)


def _fit(features: np.ndarray, labels: list[str], max_iter: int,
         c: float) -> LogisticRegression:
    """Multinomial LogReg on raw features (lbfgs is multinomial for multiclass)."""
    clf = LogisticRegression(max_iter=max_iter, C=c)
    clf.fit(features, labels)
    return clf


def _log_wandb(metrics: dict) -> None:
    try:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "weavehacks-soc-probes"),
            job_type="train_probes",
            config={"model_id": MODEL_ID, "layer": LAYER, "d_model": D_MODEL},
        )
        run.log(metrics)
        run.finish()
        print("[train] logged metrics to W&B")
    except Exception as exc:  # noqa: BLE001 — W&B is best-effort, never block training
        print(f"[train] W&B logging skipped: {exc}")


def _write_config(out_dir: Path, agent_clf: LogisticRegression,
                  tool_clf: LogisticRegression) -> None:
    config = {
        "model_id": MODEL_ID,
        "layer_index": LAYER,
        "hidden_states_index": LAYER + 1,
        "token_strategy": "last_token",
        "tokenization": "raw_text_no_chat_template",
        "add_generation_prompt": False,
        "padding_side": "left",
        "max_length": MAX_LENGTH,
        "forward_dtype": "float16",
        "feature_dtype": "float32",
        "d_model": D_MODEL,
        "agent_probe_file": AGENT_PROBE_FILE,
        "tool_probe_file": TOOL_PROBE_FILE,
        "agent_classes": int(len(agent_clf.classes_)),
        "tool_classes": int(len(tool_clf.classes_)),
        "sklearn_trained_version": sklearn.__version__,
        "source_branch": "feat/agent-layer",
        "note": "Retrained on cleaned data (data/registry_clean.jsonl). Contract "
                "matches src/probes/extract_h.py: layer 24, raw tokenizer, last "
                "token. Inference must feed raw h (no scaler).",
    }
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", default="data/activations_layer24.npy")
    parser.add_argument("--labels", default="data/activation_labels.jsonl")
    parser.add_argument("--out-dir", default="data/probes")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--C", type=float, default=1.0, dest="c")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--force", action="store_true",
                        help="save probes even if the gate fails")
    args = parser.parse_args()

    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) < (1, 9):
        print(f"WARNING: scikit-learn {sklearn.__version__} < 1.9 — saved probes "
              "may crash in predict_proba. Install scikit-learn>=1.9.0.",
              file=sys.stderr)

    activations_path = Path(args.activations)
    labels_path = Path(args.labels)
    for path in (activations_path, labels_path):
        if not path.exists():
            print(f"ERROR: {path} not found. Run "
                  "`modal run modal_app/extract_activations.py::batch` first.",
                  file=sys.stderr)
            return 1

    features = np.load(activations_path).astype(np.float32)
    rows = _load_labels(labels_path)
    if len(features) != len(rows):
        print(f"ERROR: activations ({len(features)}) and labels ({len(rows)}) "
              "row counts differ — they must be index-aligned.", file=sys.stderr)
        return 1
    print(f"[train] loaded activations {features.shape}, labels {len(rows)}")

    splits = _split_indices(rows, args.seed, args.val_frac, args.test_frac)
    train_idx, val_idx = splits["train"], splits.get("val", [])
    if not val_idx:
        print("ERROR: no validation rows after split", file=sys.stderr)
        return 1
    print(f"[train] split: train={len(train_idx)} val={len(val_idx)} "
          f"test={len(splits.get('test', []))}")

    x_train = features[train_idx]
    x_val = features[val_idx]
    agent_train = [rows[i]["agent_id"] for i in train_idx]
    tool_train = [rows[i]["tool_id"] for i in train_idx]
    agent_val = [rows[i]["agent_id"] for i in val_idx]
    tool_val = [rows[i]["tool_id"] for i in val_idx]

    print("[train] fitting agent probe...")
    agent_clf = _fit(x_train, agent_train, args.max_iter, args.c)
    print("[train] fitting tool probe (200-way, may take a minute)...")
    tool_clf = _fit(x_train, tool_train, args.max_iter, args.c)

    agent_proba = agent_clf.predict_proba(x_val)
    tool_proba = tool_clf.predict_proba(x_val)
    tool_to_agent = _tool_to_agent()

    metrics = {
        "agent_top1": _recall_at_k(agent_proba, agent_clf.classes_, agent_val, 1),
        "agent_recall_at_2": _recall_at_k(agent_proba, agent_clf.classes_,
                                          agent_val, 2),
        "tool_top1": _recall_at_k(tool_proba, tool_clf.classes_, tool_val, 1),
        "tool_recall_at_5_masked": _masked_tool_recall_at_5(
            tool_proba, tool_clf.classes_, tool_val, agent_val, tool_to_agent),
        "agent_classes": int(len(agent_clf.classes_)),
        "tool_classes": int(len(tool_clf.classes_)),
    }
    print("[train] validation metrics:")
    for key, value in metrics.items():
        print(f"  {key:<24}: {value:.4f}" if isinstance(value, float)
              else f"  {key:<24}: {value}")

    _log_wandb(metrics)

    agent_ok = metrics["agent_recall_at_2"] >= AGENT_RECALL_AT_2_GATE
    tool_ok = metrics["tool_recall_at_5_masked"] >= TOOL_RECALL_AT_5_GATE
    passed = agent_ok and tool_ok
    print(f"[train] GATE: agent recall@2 {'PASS' if agent_ok else 'FAIL'} "
          f"(>={AGENT_RECALL_AT_2_GATE}) · tool recall@5 "
          f"{'PASS' if tool_ok else 'FAIL'} (>={TOOL_RECALL_AT_5_GATE})")

    if not passed and not args.force:
        print("[train] gate failed — NOT writing probes (use --force to override). "
              "Debug tokenization/layer parity or run a layer sweep before LogReg "
              "changes (mi_implementation.md §8).", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(agent_clf, out_dir / AGENT_PROBE_FILE)
    joblib.dump(tool_clf, out_dir / TOOL_PROBE_FILE)
    _write_config(out_dir, agent_clf, tool_clf)
    print(f"[train] wrote {AGENT_PROBE_FILE}, {TOOL_PROBE_FILE}, {CONFIG_FILE} "
          f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
