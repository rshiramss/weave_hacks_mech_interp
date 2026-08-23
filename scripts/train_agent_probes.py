"""Train per-agent tool probes plus a global orchestrator probe.

Instead of one global 200-class tool probe, this trains one small tool probe
per agent (~40 classes each) on top of a 5-class orchestrator probe, then
evaluates the full two-stage routing pipeline.

Uses the cached layer-24 hidden states (no model loading / GPU). CPU-only,
numpy + sklearn + joblib.

Run from the project root:
    python scripts/train_agent_probes.py
"""

import json
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Each dataset is (cache .npy, query .json, optional clean-indices .json). The
# clean-indices file, when present, lists the original cache rows that survived
# query cleaning and is used to re-slice the cache back into alignment. Pass
# None for datasets whose query file was left untouched (e.g. descriptive).
DATASETS = [
    (
        "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy",
        "data/queries_mixed.json",
        "data/clean_indices_mixed.json",
    ),
    (
        "data/probe_cache/nl_Qwen2.5-7B-Instruct_layer24.npy",
        "data/queries_nl.json",
        "data/clean_indices_nl.json",
    ),
    (
        "data/probe_cache/descriptive_Qwen2.5-7B-Instruct_layer24.npy",
        "data/queries_descriptive.json",
        None,
    ),
]
REGISTRY_PATH = "data/registry.json"
PROBES_DIR = "data/probes"
MANIFEST_PATH = os.path.join(PROBES_DIR, "manifest.json")
SEED = 42
TOP_K = 5

PROBE_KW = dict(
    C=1.0,
    solver="saga",
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1,
)


def top_k_indices(proba, k):
    """Return indices of the top-k columns per row (unordered within the row)."""
    k = min(k, proba.shape[1])
    return np.argsort(-proba, axis=1)[:, :k]


def main():
    print("=" * 70)
    missing = [c for c, _, _ in DATASETS if not os.path.exists(c)]
    if missing:
        print("ERROR: required cache file(s) not found:")
        for m in missing:
            print(f"  - {m}")
        print("Generate these layer-24 Qwen caches before training.")
        sys.exit(1)

    print("Loading and concatenating hidden states + labels from 3 datasets...")
    X_parts = []
    rows = []
    for cache_path, query_path, indices_path in DATASETS:
        print(f"\n  dataset: {query_path}")
        Xi = np.load(cache_path)
        with open(query_path) as f:
            ri = json.load(f)
        print(f"    cache {cache_path}: {Xi.shape}")

        # Re-align the cache with the (possibly cleaned) query file by keeping
        # only the original rows recorded in the clean-indices mapping.
        if indices_path is not None:
            with open(indices_path) as f:
                keep_idx = json.load(f)
            Xi = Xi[keep_idx]
            print(f"    sliced cache via {indices_path}: {Xi.shape}")

        if Xi.shape[0] != len(ri):
            print(f"ERROR: {cache_path} rows ({Xi.shape[0]}) != "
                  f"{query_path} rows ({len(ri)}).")
            sys.exit(1)

        X_parts.append(Xi)
        rows.extend(ri)
        print(f"    {len(ri)} queries")

    X = np.vstack(X_parts)
    print(f"\n  combined X shape: {X.shape}")
    y_tool = np.array([r["tool_id"] for r in rows])
    y_agent = np.array([r["agent_id"] for r in rows])
    print(f"  {len(rows)} queries total")

    if X.shape[0] != len(rows):
        print(f"ERROR: combined cache rows ({X.shape[0]}) != "
              f"query rows ({len(rows)}).")
        sys.exit(1)

    print("Loading registry from", REGISTRY_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    agent_ids = [a["agent_id"] for a in registry["agents"]]
    tools_by_agent = {aid: [] for aid in agent_ids}
    for t in registry["tools"]:
        tools_by_agent.setdefault(t["agent_id"], []).append(t["tool_id"])
    print(f"  {len(agent_ids)} agents:")
    for aid in agent_ids:
        print(f"    {aid}: {len(tools_by_agent.get(aid, []))} tools")

    print("\nSplitting train/test (80/20, stratify by tool_id, seed=%d)..." % SEED)
    idx = np.arange(len(rows))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y_tool
    )
    X_tr, X_te = X[train_idx], X[test_idx]
    ya_tr, ya_te = y_agent[train_idx], y_agent[test_idx]
    yt_tr, yt_te = y_tool[train_idx], y_tool[test_idx]
    print(f"  train={len(train_idx)}, test={len(test_idx)}")

    os.makedirs(PROBES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: global orchestrator probe (5-class agent classifier)
    # ------------------------------------------------------------------
    print("\n[1/2] Training orchestrator probe (5-class agent)...")
    orchestrator = LogisticRegression(verbose=1, **PROBE_KW)
    orchestrator.fit(X_tr, ya_tr)
    orch_path = os.path.join(PROBES_DIR, "orchestrator_probe_best.pkl")
    joblib.dump(orchestrator, orch_path)
    orch_acc = accuracy_score(ya_te, orchestrator.predict(X_te))
    print(f"  orchestrator top-1 agent accuracy = {orch_acc:.4f}")
    print(f"  saved -> {orch_path}")

    # ------------------------------------------------------------------
    # Stage 2: one tool probe per agent
    # ------------------------------------------------------------------
    print("\n[2/2] Training per-agent tool probes...")
    tool_probes = {}
    tool_probe_paths = {}
    for i, aid in enumerate(agent_ids, 1):
        mask = ya_tr == aid
        n = int(mask.sum())
        n_classes = len(np.unique(yt_tr[mask]))
        print(f"  ({i}/{len(agent_ids)}) {aid}: {n} train rows, "
              f"{n_classes} tool classes ... training", flush=True)
        probe = LogisticRegression(**PROBE_KW)
        probe.fit(X_tr[mask], yt_tr[mask])
        path = os.path.join(PROBES_DIR, f"tool_probe_{aid}.pkl")
        joblib.dump(probe, path)
        tool_probes[aid] = probe
        tool_probe_paths[aid] = path
        print(f"      done -> {path}", flush=True)

    # ------------------------------------------------------------------
    # Evaluation: full two-stage pipeline
    # ------------------------------------------------------------------
    print("\nEvaluating two-stage pipeline on test set...")
    pred_agents = orchestrator.predict(X_te)

    total = len(test_idx)
    hits_at_k = 0
    top1_correct = 0
    # Per-agent stats keyed by the GROUND TRUTH agent.
    per_agent_total = {aid: 0 for aid in agent_ids}
    per_agent_hits = {aid: 0 for aid in agent_ids}

    for i in range(total):
        true_agent = ya_te[i]
        true_tool = yt_te[i]
        per_agent_total[true_agent] += 1

        pred_agent = pred_agents[i]
        probe = tool_probes.get(pred_agent)
        if probe is None:
            continue

        proba = probe.predict_proba(X_te[i : i + 1])[0]
        order = np.argsort(-proba)
        classes = probe.classes_

        top1_tool = classes[order[0]]
        if top1_tool == true_tool:
            top1_correct += 1

        topk_tools = set(classes[order[:TOP_K]])
        if true_tool in topk_tools:
            hits_at_k += 1
            per_agent_hits[true_agent] += 1

        if (i + 1) % 200 == 0:
            print(f"  evaluated {i + 1}/{total} "
                  f"(running recall@{TOP_K}={hits_at_k / (i + 1):.4f})",
                  flush=True)

    overall_recall = hits_at_k / total
    overall_top1 = top1_correct / total

    print("\n" + "=" * 70)
    print("TWO-STAGE PIPELINE RESULTS")
    print("=" * 70)
    print(f"Overall recall@{TOP_K}:    {overall_recall:.4f}  ({hits_at_k}/{total})")
    print(f"Overall top-1 accuracy: {overall_top1:.4f}  ({top1_correct}/{total})")
    print(f"Orchestrator agent acc: {orch_acc:.4f}")
    print("\nPer-agent recall@%d (by ground-truth agent):" % TOP_K)
    per_agent_recall = {}
    for aid in agent_ids:
        n = per_agent_total[aid]
        r = per_agent_hits[aid] / n if n else 0.0
        per_agent_recall[aid] = r
        print(f"  {aid:<18} recall@{TOP_K}={r:.4f}  ({per_agent_hits[aid]}/{n})")

    weakest = min(per_agent_recall, key=per_agent_recall.get)
    print(f"\nWeakest agent: {weakest} "
          f"(recall@{TOP_K}={per_agent_recall[weakest]:.4f})")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest = {
        "datasets": [
            {"cache": c, "queries": q, "clean_indices": idx}
            for c, q, idx in DATASETS
        ],
        "registry_path": REGISTRY_PATH,
        "seed": SEED,
        "top_k": TOP_K,
        "probe_kwargs": PROBE_KW,
        "orchestrator_probe": {
            "path": orch_path,
            "agent_ids": agent_ids,
            "test_top1_accuracy": orch_acc,
        },
        "tool_probes": {
            aid: {
                "path": tool_probe_paths[aid],
                "n_tools": len(tools_by_agent.get(aid, [])),
                "test_recall_at_%d" % TOP_K: per_agent_recall[aid],
            }
            for aid in agent_ids
        },
        "two_stage": {
            "recall_at_%d" % TOP_K: overall_recall,
            "top1_accuracy": overall_top1,
        },
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved -> {MANIFEST_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
