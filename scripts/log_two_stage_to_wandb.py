"""Log the improved two-stage probe run to W&B + Weave (logging only, no training).

Compares three approaches on the SAME held-out test split (seed=42, layer 24,
all three datasets):

  * Naive (Llama-3.1-8B, all 200 tools in context) -- reference only. Naive needs
    the GPU LLM, so (per this repo's convention in weave_evaluation.py) its
    per-query decisions are stubbed and its headline number is the prior
    benchmark E2E accuracy from results.txt, clearly labelled.
  * RAG (all-MiniLM-L6-v2, top-5 cosine) -- recomputed here on the test set.
  * Two-stage probe (orchestrator agent probe -> per-agent tool probe) -- the
    new improved router, loaded from data/probes/ and recomputed here.

Outputs:
  * One W&B run ("two-stage-vs-rag-vs-naive", group "full-benchmark") with a
    comparison wandb.Table, per-approach summary metrics, a per-query predictions
    table, and recall@5 / top-1 bar charts.
  * Weave per-query traces: one Evaluation over a stratified sample of the test
    set, with RAG / Two-stage real predictions and Naive stub, scored for tool
    correctness, top-5 hit, agent correctness, latency, and tokens.

Credentials follow the repo pattern: load_dotenv() then WANDB_KEY/WANDB_API_KEY.

Run from the project root:
    python scripts/log_two_stage_to_wandb.py
"""

import asyncio
import json
import os
import time

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

import wandb
import weave

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS = [
    ("data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_mixed.json"),
    ("data/probe_cache/nl_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_nl.json"),
    ("data/probe_cache/descriptive_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_descriptive.json"),
]
REGISTRY_PATH = "data/registry.json"
MANIFEST_PATH = "data/probes/manifest_two_stage.json"
ORCH_PATH = "data/probes/orchestrator_probe_best.pkl"
SEED = 42
TOP_K = 5
TRACE_SAMPLE = 300  # number of test queries to log as Weave traces

# Prior-benchmark naive E2E accuracy (results.txt router-sweep summary). Logged
# as a labelled reference; NOT measured on this test split.
NAIVE_PRIOR_E2E = 0.658
NAIVE_TOKENS = 4200
RAG_TOKENS = 800
PROBE_TOKENS = 50


def load_wandb_config():
    load_dotenv()
    key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
    if not key:
        raise KeyError("Set WANDB_KEY (or WANDB_API_KEY) in your environment/.env")
    os.environ["WANDB_API_KEY"] = key
    entity = os.environ.get("WANDB_ENTITY", "abrahambhatti525-santa-clara-university")
    project = os.environ.get("WANDB_PROJECT", "ToolOptim")
    return key, entity, project


# ---------------------------------------------------------------------------
# Data + probes
# ---------------------------------------------------------------------------

def load_everything():
    X_parts, rows = [], []
    for cache_path, query_path in DATASETS:
        Xi = np.load(cache_path)
        with open(query_path) as f:
            ri = json.load(f)
        assert Xi.shape[0] == len(ri), f"{cache_path} vs {query_path} misaligned"
        X_parts.append(Xi)
        rows.extend(ri)
    X = np.vstack(X_parts)
    y_tool = np.array([r["tool_id"] for r in rows])
    y_agent = np.array([r["agent_id"] for r in rows])
    qtext = np.array([r["query_text"] for r in rows], dtype=object)

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    agent_ids = [a["agent_id"] for a in registry["agents"]]
    tool_ids = [t["tool_id"] for t in registry["tools"]]
    tool_desc = {t["tool_id"]: t["description"] for t in registry["tools"]}
    tool_to_agent = {t["tool_id"]: t["agent_id"] for t in registry["tools"]}

    orchestrator = joblib.load(ORCH_PATH)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    tool_probes = {aid: joblib.load(manifest["tool_probes"][aid]["path"])
                   for aid in agent_ids}
    return (X, y_tool, y_agent, qtext, registry, agent_ids, tool_ids,
            tool_desc, tool_to_agent, orchestrator, tool_probes, manifest)


def probe_classes(p):
    return p.named_steps["logisticregression"].classes_


# ---------------------------------------------------------------------------
# Full-test metric computation (fresh, not copied from the manifest)
# ---------------------------------------------------------------------------

def compute_metrics(X, y_tool, y_agent, qtext, te, agent_ids, tool_ids,
                    tool_desc, orchestrator, tool_probes):
    yt = y_tool[te]
    ya = y_agent[te]
    n = len(te)

    # ---- two-stage probe ----
    agent_pred = orchestrator.predict(X[te])
    probe_top5 = [None] * n
    probe_top1 = [None] * n
    probe_agent = list(agent_pred)
    hit5 = np.zeros(n, bool)
    hit1 = np.zeros(n, bool)
    for aid in agent_ids:
        probe = tool_probes[aid]
        classes = probe_classes(probe)
        mp = np.where(agent_pred == aid)[0]
        if len(mp):
            proba = probe.predict_proba(X[te][mp])
            order = np.argsort(-proba, axis=1)
            for k, ridx in enumerate(mp):
                t5 = list(classes[order[k, :TOP_K]])
                probe_top5[ridx] = t5
                probe_top1[ridx] = t5[0]
                hit5[ridx] = yt[ridx] in t5
                hit1[ridx] = t5[0] == yt[ridx]
    probe_metrics = {"recall@5": float(hit5.mean()), "top1": float(hit1.mean())}

    # ---- RAG (MiniLM top-5 cosine) ----
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    tool_emb = emb.encode([tool_desc[t] for t in tool_ids],
                          normalize_embeddings=True, show_progress_bar=False)
    q_emb = emb.encode(list(qtext[te]), normalize_embeddings=True,
                       batch_size=256, show_progress_bar=False)
    sims = q_emb @ tool_emb.T
    order = np.argsort(-sims, axis=1)
    tool_ids_arr = np.array(tool_ids)
    rag_top5 = [list(tool_ids_arr[order[i, :TOP_K]]) for i in range(n)]
    rag_top1 = [r[0] for r in rag_top5]
    rag_hit5 = np.mean([yt[i] in rag_top5[i] for i in range(n)])
    rag_hit1 = np.mean([rag_top1[i] == yt[i] for i in range(n)])
    rag_metrics = {"recall@5": float(rag_hit5), "top1": float(rag_hit1)}

    per_query = {
        "yt": yt, "ya": ya,
        "probe_top5": probe_top5, "probe_top1": probe_top1, "probe_agent": probe_agent,
        "rag_top5": rag_top5, "rag_top1": rag_top1,
    }
    return probe_metrics, rag_metrics, per_query


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def log_wandb(entity, project, probe_m, rag_m, manifest, per_query, te, qtext, n_test):
    run = wandb.init(
        entity=entity, project=project,
        name="two-stage-vs-rag-vs-naive",
        group="full-benchmark",
        job_type="comparison",
        tags=["full_benchmark", "comparison", "two_stage", "probe", "rag", "naive"],
        config={"layer": 24, "seed": SEED, "top_k": TOP_K,
                "n_test": n_test, "datasets": [c for c, _ in DATASETS]},
        reinit=True,
    )

    cols = ["Approach", "Recall@5", "Top-1", "Tokens/query", "Notes"]
    table = wandb.Table(columns=cols)
    table.add_data("Naive (Llama-3.1-8B)", None, NAIVE_PRIOR_E2E, NAIVE_TOKENS,
                   "reference: prior benchmark E2E (separate test set)")
    table.add_data("RAG (MiniLM top-5)", rag_m["recall@5"], rag_m["top1"], RAG_TOKENS,
                   f"this test set (n={n_test})")
    table.add_data("Two-stage probe (Qwen L24)", probe_m["recall@5"], probe_m["top1"],
                   PROBE_TOKENS, f"this test set (n={n_test})")

    # per-approach summary metrics
    run.summary.update({
        "naive/top1_prior": NAIVE_PRIOR_E2E,
        "rag/recall@5": rag_m["recall@5"], "rag/top1": rag_m["top1"],
        "two_stage/recall@5": probe_m["recall@5"], "two_stage/top1": probe_m["top1"],
        "two_stage/oracle_recall@5": manifest["two_stage"]["oracle_recall_at_5"],
        "two_stage/orchestrator_agent_acc": manifest["orchestrator"]["test_agent_acc"],
    })

    # per-query predictions table (sampled to keep it light)
    pq_cols = ["query_text", "true_tool", "true_agent", "probe_pred_agent",
               "probe_top1", "probe_hit@5", "rag_top1", "rag_hit@5"]
    pq_table = wandb.Table(columns=pq_cols)
    yt, ya = per_query["yt"], per_query["ya"]
    sample = range(min(500, len(te)))
    for i in sample:
        pq_table.add_data(
            str(qtext[te][i]), str(yt[i]), str(ya[i]),
            str(per_query["probe_agent"][i]), str(per_query["probe_top1"][i]),
            bool(yt[i] in per_query["probe_top5"][i]),
            str(per_query["rag_top1"][i]),
            bool(yt[i] in per_query["rag_top5"][i]),
        )

    recall_bar = wandb.Table(
        data=[["RAG (MiniLM top-5)", rag_m["recall@5"]],
              ["Two-stage probe", probe_m["recall@5"]]],
        columns=["Approach", "Recall@5"])
    top1_bar = wandb.Table(
        data=[["Naive (prior)", NAIVE_PRIOR_E2E],
              ["RAG (MiniLM top-5)", rag_m["top1"]],
              ["Two-stage probe", probe_m["top1"]]],
        columns=["Approach", "Top-1"])

    wandb.log({
        "approach_comparison": table,
        "per_query_predictions": pq_table,
        "recall5_by_approach": wandb.plot.bar(recall_bar, "Approach", "Recall@5",
                                              title="Recall@5 by Approach"),
        "top1_by_approach": wandb.plot.bar(top1_bar, "Approach", "Top-1",
                                           title="Top-1 by Approach"),
    })
    url = run.url
    wandb.finish()
    print(f"W&B run logged: {url}")
    return url


# ---------------------------------------------------------------------------
# Weave traces
# ---------------------------------------------------------------------------

def log_weave_traces(entity, project, X, qtext, te, per_query, sample_idx,
                     tool_desc, tool_ids, tool_to_agent, orchestrator, tool_probes,
                     agent_ids):
    weave.init(f"{entity}/{project}")

    # Map query_text -> cache row for the sampled test rows so the probe model
    # can fetch the exact hidden-state vector it was evaluated on.
    row_for_query = {}
    rows = []
    for pos in sample_idx:
        gidx = int(te[pos])
        qt = str(qtext[gidx])
        row_for_query[qt] = gidx
        rows.append({"query_text": qt,
                     "tool_id": str(per_query["yt"][pos]),
                     "agent_id": str(per_query["ya"][pos])})

    dataset = weave.Dataset(name="two_stage_test_sample", rows=rows)
    weave.publish(dataset)

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tool_emb = embedder.encode([tool_desc[t] for t in tool_ids],
                               normalize_embeddings=True, show_progress_bar=False)
    tool_ids_arr = np.array(tool_ids)

    @weave.op
    def tool_correct(tool_id: str, output: dict) -> dict:
        return {"correct": output["predicted_tool"] == tool_id}

    @weave.op
    def tool_in_top5(tool_id: str, output: dict) -> dict:
        return {"hit": tool_id in output.get("top5_tools", [])}

    @weave.op
    def agent_correct(agent_id: str, output: dict) -> dict:
        return {"correct": output["predicted_agent"] == agent_id}

    @weave.op
    def latency_ms(output: dict) -> dict:
        return {"ms": output["latency_ms"]}

    @weave.op
    def tokens_used(output: dict) -> dict:
        return {"tokens": output["token_count"]}

    class NaiveModel(weave.Model):
        name: str = "Naive-Llama-3.1-8B"

        @weave.op
        def predict(self, query_text: str) -> dict:
            # Needs the GPU LLM; stubbed per repo convention (benchmarked
            # separately in full_benchmark.py).
            return {"predicted_tool": "stub", "predicted_agent": "stub",
                    "top5_tools": [], "token_count": NAIVE_TOKENS, "latency_ms": 0}

    class RAGModel(weave.Model):
        name: str = "RAG-MiniLM-top5"

        @weave.op
        def predict(self, query_text: str) -> dict:
            t0 = time.perf_counter()
            q = embedder.encode([query_text], normalize_embeddings=True)[0]
            sims = tool_emb @ q
            top5 = list(tool_ids_arr[np.argsort(-sims)[:TOP_K]])
            ms = (time.perf_counter() - t0) * 1000
            return {"predicted_tool": top5[0],
                    "predicted_agent": tool_to_agent.get(top5[0], "unknown"),
                    "top5_tools": [str(t) for t in top5],
                    "token_count": RAG_TOKENS, "latency_ms": ms}

    class TwoStageProbeModel(weave.Model):
        name: str = "Two-Stage-Probe-Qwen-L24"

        @weave.op
        def predict(self, query_text: str) -> dict:
            gidx = row_for_query.get(query_text)
            if gidx is None:
                return {"predicted_tool": "stub", "predicted_agent": "stub",
                        "top5_tools": [], "token_count": PROBE_TOKENS, "latency_ms": 0}
            x = X[gidx:gidx + 1]
            t0 = time.perf_counter()
            agent = str(orchestrator.predict(x)[0])
            probe = tool_probes[agent]
            classes = probe_classes(probe)
            proba = probe.predict_proba(x)[0]
            top5 = list(classes[np.argsort(-proba)[:TOP_K]])
            ms = (time.perf_counter() - t0) * 1000
            return {"predicted_tool": str(top5[0]),
                    "predicted_agent": agent,
                    "top5_tools": [str(t) for t in top5],
                    "token_count": PROBE_TOKENS, "latency_ms": ms}

    evaluation = weave.Evaluation(
        name="two_stage_vs_rag_vs_naive",
        dataset=dataset,
        scorers=[tool_correct, tool_in_top5, agent_correct, latency_ms, tokens_used],
    )
    print(f"Logging Weave traces for {len(rows)} test queries x 3 models ...")
    asyncio.run(evaluation.evaluate(NaiveModel()))
    asyncio.run(evaluation.evaluate(RAGModel()))
    asyncio.run(evaluation.evaluate(TwoStageProbeModel()))
    print("Weave traces logged.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    key, entity, project = load_wandb_config()
    wandb.login(key=key)

    (X, y_tool, y_agent, qtext, registry, agent_ids, tool_ids, tool_desc,
     tool_to_agent, orchestrator, tool_probes, manifest) = load_everything()

    # Reproduce the exact test split used in training.
    idx = np.arange(len(y_tool))
    _, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_tool)
    n_test = len(te)
    print(f"Test set: {n_test} queries.")

    print("Computing fresh full-test metrics (probe + RAG)...")
    probe_m, rag_m, per_query = compute_metrics(
        X, y_tool, y_agent, qtext, te, agent_ids, tool_ids, tool_desc,
        orchestrator, tool_probes)
    print(f"  Two-stage: recall@5={probe_m['recall@5']:.4f} top1={probe_m['top1']:.4f}")
    print(f"  RAG:       recall@5={rag_m['recall@5']:.4f} top1={rag_m['top1']:.4f}")
    print(f"  Naive:     top1(prior ref)={NAIVE_PRIOR_E2E}")

    log_wandb(entity, project, probe_m, rag_m, manifest, per_query, te, qtext, n_test)

    # stratified-ish sample for Weave traces
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(n_test, size=min(TRACE_SAMPLE, n_test), replace=False)
    log_weave_traces(entity, project, X, qtext, te, per_query, sample_idx,
                     tool_desc, tool_ids, tool_to_agent, orchestrator, tool_probes,
                     agent_ids)

    print("Done. Comparison + per-query traces logged to W&B project "
          f"'{project}'.")


if __name__ == "__main__":
    main()
