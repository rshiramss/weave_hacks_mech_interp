"""Local hard-set eval: two-stage probe vs RAG vs true naive Llama.

No W&B / Weave logging. This uses the same seed-42 held-out 20% split as the
training scripts, then filters to queries with no obvious lexical cue from the
ground-truth tool id or the first few description words. It samples a small
balanced subset so true-naive Llama stays quick.

Run from the project root:
    python scripts/eval_hard_local.py
"""

import json
import os
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


def has_obvious_cue(text, tool_id, tool_desc):
    text = text.lower()
    phrase = tool_id.replace("_", " ")
    tool_tokens = [t for t in tool_id.split("_") if len(t) > 3]
    desc_tokens = [
        w.strip(".,:;()[]").lower()
        for w in tool_desc[tool_id].split()
        if len(w.strip(".,:;()[]")) > 5
    ]
    return (
        phrase in text
        or any(tok in text for tok in tool_tokens)
        or any(tok in text for tok in desc_tokens[:4])
    )


def probe_classes(probe):
    return probe.named_steps["logisticregression"].classes_


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
    X = np.vstack(X_parts)
    return X, rows, np.array(sources)


def topk_from_probe(probe, x):
    classes = probe_classes(probe)
    proba = probe.predict_proba(x)[0]
    return [str(t) for t in classes[np.argsort(-proba)[:TOP_K]]]


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
    hard_idx = [
        i for i in test_idx
        if not has_obvious_cue(str(qtext[i]), str(y_tool[i]), tool_desc)
    ]

    rng = np.random.default_rng(SEED)
    sample = []
    for aid in agent_ids:
        pool = np.array([i for i in hard_idx if y_agent[i] == aid])
        take = min(PER_AGENT, len(pool))
        sample.extend(rng.choice(pool, size=take, replace=False))
    sample = np.array(sample)

    print(f"Total held-out test rows: {len(test_idx)}")
    print(f"Hard/no-cue held-out rows: {len(hard_idx)}")
    print(f"Sampled hard eval rows: {len(sample)} ({PER_AGENT}/agent target)")
    for aid in agent_ids:
        print(f"  {aid:<18} n={(y_agent[sample] == aid).sum()}")

    print("\nBuilding RAG index...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tool_emb = embedder.encode(
        [tool_desc[t] for t in tool_ids],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    tool_ids_arr = np.array(tool_ids)

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
    mistakes = []

    print("\nRunning local hard eval...")
    for n, i in enumerate(sample, 1):
        q = str(qtext[i])
        gt_tool = str(y_tool[i])
        gt_agent = str(y_agent[i])
        x = X[i:i + 1]

        # Two-stage probe
        t0 = time.perf_counter()
        pred_agent = str(orchestrator.predict(x)[0])
        top5 = topk_from_probe(tool_probes[pred_agent], x)
        dt = (time.perf_counter() - t0) * 1000
        results["probe"]["top1"] += top5[0] == gt_tool
        results["probe"]["r5"] += gt_tool in top5
        results["probe"]["agent"] += pred_agent == gt_agent
        results["probe"]["times"].append(dt)

        # RAG
        t0 = time.perf_counter()
        q_emb = embedder.encode([q], normalize_embeddings=True)[0]
        sims = tool_emb @ q_emb
        rag_top5 = [str(t) for t in tool_ids_arr[np.argsort(-sims)[:TOP_K]]]
        dt = (time.perf_counter() - t0) * 1000
        rag_agent = tool_to_agent.get(rag_top5[0], "unknown")
        results["rag"]["top1"] += rag_top5[0] == gt_tool
        results["rag"]["r5"] += gt_tool in rag_top5
        results["rag"]["agent"] += rag_agent == gt_agent
        results["rag"]["times"].append(dt)

        # True naive Llama
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
        results["naive"]["r5"] += naive_pred == gt_tool  # single output
        results["naive"]["agent"] += naive_agent == gt_agent
        results["naive"]["times"].append(dt)

        if len(mistakes) < 12 and (
            top5[0] != gt_tool or rag_top5[0] != gt_tool or naive_pred != gt_tool
        ):
            mistakes.append({
                "q": q,
                "gt": gt_tool,
                "probe": top5[0],
                "rag": rag_top5[0],
                "naive": naive_pred,
            })

        print(
            f"[{n:02d}/{len(sample)}] gt={gt_tool} "
            f"probe={top5[0]} rag={rag_top5[0]} naive={naive_pred}",
            flush=True,
        )

    total = len(sample)
    print("\n================ HARD LOCAL RESULTS ================")
    print(f"Hard filter: held-out test rows with no tool-id/description lexical cue")
    print(f"n={total}, all from held-out 20% split, no W&B logging")
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

    print("\nExample disagreements/misses:")
    for m in mistakes:
        print(f"- q: {m['q']}")
        print(f"  gt={m['gt']} | probe={m['probe']} | rag={m['rag']} | naive={m['naive']}")


if __name__ == "__main__":
    main()
