"""Train a strong two-stage tool router and compare it to RAG.

Stage 1: global orchestrator probe (agent, 5-class) -- already strong.
Stage 2: one tool probe per agent (~40-class), routed by the predicted agent.

Key improvements over the earlier per-agent run:
  * StandardScaler + lbfgs (converges fast and fully; saga was hitting max_iter
    and under-training the probes).
  * All three aligned datasets (mixed + nl + descriptive, 17k rows).
  * Small per-agent C sweep, selected on a held-out validation split (no test
    leakage), then refit on the full training portion.

Also measures RAG (all-MiniLM-L6-v2, top-5 cosine) recall@5 / top-1 on the
exact same test split so the comparison is apples-to-apples.

CPU-only. numpy + sklearn + joblib (+ sentence-transformers for the RAG row).

Run from the project root:
    python scripts/train_two_stage_best.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASETS = [
    ("data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_mixed.json"),
    ("data/probe_cache/nl_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_nl.json"),
    ("data/probe_cache/descriptive_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_descriptive.json"),
]
REGISTRY_PATH = "data/registry.json"
PROBES_DIR = "data/probes"
MANIFEST_PATH = os.path.join(PROBES_DIR, "manifest_two_stage.json")
SEED = 42
TOP_K = 5
C_GRID = [0.5, 1.0, 2.0, 5.0]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def recall_at_k_from_proba(proba, classes, y_true, k):
    topk = classes[np.argsort(-proba, axis=1)[:, :k]]
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])


def load_data():
    X_parts, rows = [], []
    for cache_path, query_path in DATASETS:
        Xi = np.load(cache_path)
        with open(query_path) as f:
            ri = json.load(f)
        if Xi.shape[0] != len(ri):
            log(f"ERROR: {cache_path} ({Xi.shape[0]}) != {query_path} ({len(ri)})")
            sys.exit(1)
        X_parts.append(Xi)
        rows.extend(ri)
    X = np.vstack(X_parts)
    if X.shape[0] != len(rows):
        log(f"ERROR: combined rows mismatch {X.shape[0]} != {len(rows)}")
        sys.exit(1)
    y_tool = np.array([r["tool_id"] for r in rows])
    y_agent = np.array([r["agent_id"] for r in rows])
    qtext = np.array([r["query_text"] for r in rows], dtype=object)
    return X, y_tool, y_agent, qtext


def make_probe(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                           class_weight="balanced"),
    )


def main():
    t_start = time.time()
    os.makedirs(PROBES_DIR, exist_ok=True)

    log("Loading 3 aligned datasets...")
    X, y_tool, y_agent, qtext = load_data()
    log(f"  X={X.shape}, tools={len(set(y_tool))}, agents={sorted(set(y_agent))}")

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    agent_ids = [a["agent_id"] for a in registry["agents"]]

    # ---- split: train_full / test, then train / val from train_full ----
    idx = np.arange(len(y_tool))
    tr_full, te = train_test_split(idx, test_size=0.2, random_state=SEED,
                                   stratify=y_tool)
    tr, va = train_test_split(tr_full, test_size=0.15, random_state=SEED,
                              stratify=y_tool[tr_full])
    log(f"  split: train={len(tr)} val={len(va)} test={len(te)}")

    # ---------------- Stage 1: orchestrator (agent) ----------------
    log("Training orchestrator (agent) probe...")
    best_orch, best_orch_c, best_orch_acc = None, None, -1.0
    for C in (0.5, 1.0):
        p = make_probe(C).fit(X[tr], y_agent[tr])
        acc = p.score(X[va], y_agent[va])
        log(f"  orchestrator C={C}: val agent acc={acc:.4f}")
        if acc > best_orch_acc:
            best_orch, best_orch_c, best_orch_acc = p, C, acc
    orchestrator = make_probe(best_orch_c).fit(X[tr_full], y_agent[tr_full])
    orch_test_acc = orchestrator.score(X[te], y_agent[te])
    joblib.dump(orchestrator, os.path.join(PROBES_DIR, "orchestrator_probe_best.pkl"))
    log(f"  orchestrator best C={best_orch_c}, TEST agent acc={orch_test_acc:.4f}")

    # ---------------- Stage 2: per-agent tool probes ----------------
    tool_probes, tool_paths, per_agent_C = {}, {}, {}
    for i, aid in enumerate(agent_ids, 1):
        m_tr = y_agent[tr] == aid
        m_va = y_agent[va] == aid
        m_full = y_agent[tr_full] == aid
        n_cls = len(set(y_tool[tr_full][m_full]))
        log(f"({i}/{len(agent_ids)}) {aid}: {m_full.sum()} train rows, "
            f"{n_cls} tool classes -- C sweep...")
        best_c, best_r5 = C_GRID[0], -1.0
        for C in C_GRID:
            p = make_probe(C).fit(X[tr][m_tr], y_tool[tr][m_tr])
            proba = p.predict_proba(X[va][m_va])
            r5 = recall_at_k_from_proba(proba, p.named_steps["logisticregression"].classes_,
                                        y_tool[va][m_va], TOP_K)
            if r5 > best_r5:
                best_c, best_r5 = C, r5
        # refit best C on full train portion for this agent
        probe = make_probe(best_c).fit(X[tr_full][m_full], y_tool[tr_full][m_full])
        path = os.path.join(PROBES_DIR, f"tool_probe_{aid}.pkl")
        joblib.dump(probe, path)
        tool_probes[aid] = probe
        tool_paths[aid] = path
        per_agent_C[aid] = best_c
        log(f"    best C={best_c} (val recall@{TOP_K}={best_r5:.4f}) -> saved")

    # ---------------- Two-stage evaluation on test ----------------
    log("Evaluating two-stage pipeline on test set...")
    agent_pred = orchestrator.predict(X[te])
    yt_te = y_tool[te]
    ya_te = y_agent[te]

    n = len(te)
    hit5 = np.zeros(n, bool)
    hit1 = np.zeros(n, bool)
    oracle5 = np.zeros(n, bool)  # routing with TRUE agent (ceiling)
    # group test rows by predicted / true agent to batch predict_proba
    for aid in agent_ids:
        probe = tool_probes[aid]
        classes = probe.named_steps["logisticregression"].classes_
        # predicted-agent routing
        mp = agent_pred == aid
        if mp.any():
            proba = probe.predict_proba(X[te][mp])
            order = np.argsort(-proba, axis=1)
            top5 = classes[order[:, :TOP_K]]
            top1 = classes[order[:, 0]]
            yt = yt_te[mp]
            hit5[mp] = [yt[j] in top5[j] for j in range(len(yt))]
            hit1[mp] = top1 == yt
        # true-agent routing (oracle ceiling)
        mt = ya_te == aid
        if mt.any():
            proba = probe.predict_proba(X[te][mt])
            top5 = classes[np.argsort(-proba, axis=1)[:, :TOP_K]]
            yt = yt_te[mt]
            oracle5[mt] = [yt[j] in top5[j] for j in range(len(yt))]

    two_stage_r5 = hit5.mean()
    two_stage_t1 = hit1.mean()
    oracle_r5 = oracle5.mean()

    # ---------------- RAG comparison on same test set ----------------
    rag_r5 = rag_t1 = None
    try:
        from sentence_transformers import SentenceTransformer
        log("Measuring RAG (all-MiniLM-L6-v2) on the same test set...")
        tool_ids = [t["tool_id"] for t in registry["tools"]]
        tool_desc = [t["description"] for t in registry["tools"]]
        emb = SentenceTransformer("all-MiniLM-L6-v2")
        tool_emb = emb.encode(tool_desc, normalize_embeddings=True,
                              show_progress_bar=False)
        q_emb = emb.encode(list(qtext[te]), normalize_embeddings=True,
                           batch_size=256, show_progress_bar=False)
        sims = q_emb @ tool_emb.T
        order = np.argsort(-sims, axis=1)
        tool_ids_arr = np.array(tool_ids)
        rag_top5 = tool_ids_arr[order[:, :TOP_K]]
        rag_top1 = tool_ids_arr[order[:, 0]]
        rag_r5 = np.mean([yt_te[j] in rag_top5[j] for j in range(n)])
        rag_t1 = np.mean(rag_top1 == yt_te)
    except Exception as e:
        log(f"  [warn] RAG measurement skipped: {type(e).__name__}: {e}")

    # ---------------- Report ----------------
    print("\n" + "=" * 64)
    print("TWO-STAGE PROBE vs RAG  (test set, layer 24, all 3 datasets)")
    print("=" * 64)
    print(f"Orchestrator agent acc : {orch_test_acc:.4f}")
    print(f"Two-stage Recall@{TOP_K}    : {two_stage_r5:.4f}")
    print(f"Two-stage Top-1        : {two_stage_t1:.4f}")
    print(f"  (oracle routing R@{TOP_K}: {oracle_r5:.4f}  <- ceiling if agent always right)")
    if rag_r5 is not None:
        print(f"RAG Recall@{TOP_K}         : {rag_r5:.4f}")
        print(f"RAG Top-1              : {rag_t1:.4f}")
        verdict = "BEATS RAG" if two_stage_r5 > rag_r5 else "below RAG"
        print(f"\n>>> Two-stage recall@{TOP_K} {verdict} "
              f"({two_stage_r5:.4f} vs {rag_r5:.4f})")
    else:
        print("RAG: skipped (see warning above); known prior baseline ~0.61 E2E")

    print(f"\nPer-agent (test, true-agent routing):")
    for aid in agent_ids:
        mt = ya_te == aid
        print(f"  {aid:<18} n={int(mt.sum()):4}  C={per_agent_C[aid]}  "
              f"recall@{TOP_K}={oracle5[mt].mean():.4f}")

    manifest = {
        "datasets": [c for c, _ in DATASETS],
        "layer": 24,
        "seed": SEED,
        "scaler": "StandardScaler",
        "solver": "lbfgs",
        "orchestrator": {"C": best_orch_c, "test_agent_acc": float(orch_test_acc),
                         "path": "data/probes/orchestrator_probe_best.pkl"},
        "tool_probes": {aid: {"C": per_agent_C[aid], "path": tool_paths[aid]}
                        for aid in agent_ids},
        "two_stage": {"recall_at_5": float(two_stage_r5),
                      "top1": float(two_stage_t1),
                      "oracle_recall_at_5": float(oracle_r5)},
        "rag": ({"recall_at_5": float(rag_r5), "top1": float(rag_t1)}
                if rag_r5 is not None else None),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"Saved manifest -> {MANIFEST_PATH}")
    log(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
