"""Local curated stress-slice eval: two-stage probe vs RAG vs true naive.

No W&B / Weave logging.

This is NOT an unbiased benchmark. It intentionally selects a held-out slice
where:
  * the exact full tool-name phrase is not present in the query,
  * the two-stage probe is top-1 correct,
  * RAG top-1 is wrong.

Then it runs true naive Llama-3.1-8B with all 200 full tool descriptions on the
same selected rows. This is useful as a demo/diagnostic slice for cases where
the probe captures a routing signal that retrieval-style matching misses.

Run from the project root:
    python scripts/eval_curated_probe_slice_local.py
"""

import json
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

DATASETS = [
    ("mixed", "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_mixed.json"),
    ("nl", "data/probe_cache/nl_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_nl.json"),
    ("descriptive", "data/probe_cache/descriptive_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_descriptive.json"),
]
REGISTRY_PATH = "data/registry.json"
MANIFEST_PATH = "data/probes/manifest_two_stage.json"
ORCH_PATH = "data/probes/orchestrator_probe_best.pkl"
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42
TOP_K = 5
PER_AGENT = 10

ROUTER_SYSTEM = (
    "You are a tool router for a SOC. Given a user query and a list of available "
    "tools, return ONLY the tool_id of the single best tool as a JSON string, "
    'e.g. "search_auth_logs". No explanation, no markdown.'
)


def format_tools(tool_ids, tool_desc):
    return "\n".join(f"- {tid}: {tool_desc[tid]}" for tid in tool_ids)


def parse_tool(text, valid_sorted):
    text = text.strip()
    try:
        v = json.loads(text)
        if isinstance(v, str) and v.strip() in valid_sorted:
            return v.strip()
        if isinstance(v, dict):
            for val in v.values():
                if isinstance(val, str) and val.strip() in valid_sorted:
                    return val.strip()
    except Exception:
        pass
    for tid in valid_sorted:
        if tid in text:
            return tid
    return None


def exact_tool_phrase(text, tool_id):
    return tool_id.replace("_", " ") in text.lower()


def probe_classes(probe):
    return probe.named_steps["logisticregression"].classes_


def topk_from_probe(probe, x):
    classes = probe_classes(probe)
    proba = probe.predict_proba(x)[0]
    return [str(t) for t in classes[np.argsort(-proba)[:TOP_K]]], float(np.max(proba))


def load_data():
    X_parts, rows, sources = [], [], []
    for name, cache_path, query_path in DATASETS:
        Xi = np.load(cache_path)
        with open(query_path) as f:
            ri = json.load(f)
        if Xi.shape[0] != len(ri):
            raise ValueError(f"{cache_path} rows != {query_path} rows")
        X_parts.append(Xi)
        rows.extend(ri)
        sources.extend([name] * len(ri))
    return np.vstack(X_parts), rows, np.array(sources)


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading data/probes...")
    X, rows, sources = load_data()
    y_tool = np.array([r["tool_id"] for r in rows])
    y_agent = np.array([r["agent_id"] for r in rows])
    qtext = np.array([r["query_text"] for r in rows], dtype=object)

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    agent_ids = [a["agent_id"] for a in registry["agents"]]
    tool_ids = [t["tool_id"] for t in registry["tools"]]
    tool_desc = {t["tool_id"]: t["description"] for t in registry["tools"]}
    tool_to_agent = {t["tool_id"]: t["agent_id"] for t in registry["tools"]}
    valid_sorted = sorted(tool_ids, key=len, reverse=True)

    orchestrator = joblib.load(ORCH_PATH)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    tool_probes = {
        aid: joblib.load(manifest["tool_probes"][aid]["path"]) for aid in agent_ids
    }

    idx = np.arange(len(rows))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_tool)

    print("Computing probe + RAG predictions on held-out test split...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tool_emb = embedder.encode(
        [tool_desc[t] for t in tool_ids],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q_emb = embedder.encode(
        list(qtext[test_idx]),
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=False,
    )
    rag_order = np.argsort(-(q_emb @ tool_emb.T), axis=1)
    tool_ids_arr = np.array(tool_ids)

    candidates_by_agent = {aid: [] for aid in agent_ids}
    for pos, i in enumerate(test_idx):
        gt_tool = str(y_tool[i])
        gt_agent = str(y_agent[i])
        if exact_tool_phrase(str(qtext[i]), gt_tool):
            continue

        pred_agent = str(orchestrator.predict(X[i:i + 1])[0])
        if pred_agent not in tool_probes:
            continue
        probe_top5, probe_conf = topk_from_probe(tool_probes[pred_agent], X[i:i + 1])
        rag_top5 = [str(t) for t in tool_ids_arr[rag_order[pos, :TOP_K]]]

        if probe_top5[0] == gt_tool and rag_top5[0] != gt_tool:
            candidates_by_agent[gt_agent].append({
                "idx": int(i),
                "probe_top5": probe_top5,
                "rag_top5": rag_top5,
                "probe_conf": probe_conf,
            })

    print("Candidate pool (held-out, no exact tool phrase, probe correct, RAG wrong):")
    total_candidates = 0
    for aid in agent_ids:
        n = len(candidates_by_agent[aid])
        total_candidates += n
        print(f"  {aid:<18} {n}")
    print(f"  total              {total_candidates}")

    rng = np.random.default_rng(SEED)
    selected = []
    for aid in agent_ids:
        pool = candidates_by_agent[aid]
        take = min(PER_AGENT, len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False)
        selected.extend(pool[int(j)] for j in chosen)

    print(f"\nSelected {len(selected)} rows ({PER_AGENT}/agent target).")

    print("Loading true naive Llama with all 200 full tool descriptions...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    if device == "cpu":
        model = model.to(device)

    all_tools_block = format_tools(tool_ids, tool_desc)
    naive_tokens = int(len(all_tools_block) / 4)
    print(f"Naive prompt has {len(tool_ids)} tools, approx {naive_tokens} tool tokens.")

    results = {
        "probe": {"top1": 0, "r5": 0, "agent": 0, "times": []},
        "rag": {"top1": 0, "r5": 0, "agent": 0, "times": []},
        "naive": {"top1": 0, "r5": 0, "agent": 0, "times": []},
    }
    rows_out = []

    print("\nRunning curated local eval...")
    for n, item in enumerate(selected, 1):
        i = item["idx"]
        q = str(qtext[i])
        gt_tool = str(y_tool[i])
        gt_agent = str(y_agent[i])
        x = X[i:i + 1]

        # Recompute probe timing/prediction on selected row.
        t0 = time.perf_counter()
        probe_agent = str(orchestrator.predict(x)[0])
        probe_top5, _ = topk_from_probe(tool_probes[probe_agent], x)
        dt = (time.perf_counter() - t0) * 1000
        results["probe"]["top1"] += probe_top5[0] == gt_tool
        results["probe"]["r5"] += gt_tool in probe_top5
        results["probe"]["agent"] += probe_agent == gt_agent
        results["probe"]["times"].append(dt)

        # RAG already computed; use selected stored top5.
        t0 = time.perf_counter()
        rag_top5 = item["rag_top5"]
        dt = (time.perf_counter() - t0) * 1000
        rag_agent = tool_to_agent.get(rag_top5[0], "unknown")
        results["rag"]["top1"] += rag_top5[0] == gt_tool
        results["rag"]["r5"] += gt_tool in rag_top5
        results["rag"]["agent"] += rag_agent == gt_agent
        results["rag"]["times"].append(dt)

        # True naive Llama.
        user = (
            f"Available tools:\n{all_tools_block}\n\nQuery: {q}\n\n"
            "Return the single best tool_id as a JSON string."
        )
        prompt = f"{ROUTER_SYSTEM}\n\n{user}"
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        dt = (time.perf_counter() - t0) * 1000
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        naive_pred = parse_tool(text, valid_sorted) or "stub"
        naive_agent = tool_to_agent.get(naive_pred, "unknown")
        results["naive"]["top1"] += naive_pred == gt_tool
        results["naive"]["r5"] += naive_pred == gt_tool
        results["naive"]["agent"] += naive_agent == gt_agent
        results["naive"]["times"].append(dt)

        rows_out.append((q, gt_tool, probe_top5[0], rag_top5[0], naive_pred))
        print(
            f"[{n:02d}/{len(selected)}] gt={gt_tool} "
            f"probe={probe_top5[0]} rag={rag_top5[0]} naive={naive_pred}",
            flush=True,
        )

    total = len(selected)
    print("\n================ CURATED PROBE-SLICE RESULTS ================")
    print("Selection: held-out rows, no exact tool phrase, probe top-1 correct, RAG top-1 wrong")
    print("This is a diagnostic/demo slice, not an unbiased benchmark. No W&B logging.")
    print(f"n={total}")
    print(f"{'Approach':<14}{'Top-1':>9}{'Recall@5':>11}{'AgentAcc':>11}{'ms/q':>10}")
    print("-" * 55)
    for name, label in [
        ("probe", "Two-stage"),
        ("rag", "RAG"),
        ("naive", "Naive"),
    ]:
        r = results[name]
        print(
            f"{label:<14}{r['top1'] / total:>9.4f}"
            f"{r['r5'] / total:>11.4f}"
            f"{r['agent'] / total:>11.4f}"
            f"{np.mean(r['times']):>10.1f}"
        )

    print("\nExamples:")
    for q, gt, probe, rag, naive in rows_out[:12]:
        print(f"- q: {q}")
        print(f"  gt={gt} | probe={probe} | rag={rag} | naive={naive}")


if __name__ == "__main__":
    main()
