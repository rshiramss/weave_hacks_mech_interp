"""3-way Weave eval (Evals tab): Two-stage probe vs RAG vs REAL naive Llama.

All three models are evaluated on the SAME small stratified sample of the
seed-42 test split, so they line up for a direct side-by-side comparison in the
W&B / Weave "Evals" tab.

  * Two-stage probe  -- orchestrator agent probe -> per-agent tool probe (cached
    layer-24 hidden states; no model load).
  * RAG              -- all-MiniLM-L6-v2 top-5 cosine.
  * Naive            -- REAL Llama-3.1-8B with all 200 tool descriptions in
    context (GPU). Kept to a small sample so it runs quickly.

Logging only -- no probe training. Run from the project root:
    python scripts/log_naive_eval.py
"""

import asyncio
import json
import os
import time

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

import weave

DATASETS = [
    ("data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_mixed.json"),
    ("data/probe_cache/nl_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_nl.json"),
    ("data/probe_cache/descriptive_Qwen2.5-7B-Instruct_layer24.npy", "data/queries_descriptive.json"),
]
REGISTRY_PATH = "data/registry.json"
MANIFEST_PATH = "data/probes/manifest_two_stage.json"
ORCH_PATH = "data/probes/orchestrator_probe_best.pkl"
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42
TOP_K = 5
SAMPLE_N = 120  # small so real-naive LLM inference stays quick

ROUTER_SYSTEM = (
    "You are a tool router for a SOC. Given a user query and a list of available "
    "tools, return ONLY the tool_id of the single best tool as a JSON string, "
    'e.g. "search_auth_logs". No explanation, no markdown.'
)

# ---- globals populated in main(), used by the weave Models ----
X = None
ROW_FOR_QUERY = {}
ORCH = None
TOOL_PROBES = {}
TOOL_IDS = []
TOOL_DESC = {}
TOOL_TO_AGENT = {}
VALID_SORTED = []
TOOL_EMB = None
EMBEDDER = None
ALL_TOOLS_BLOCK = ""
NAIVE_TOKENS = 0
_LLAMA = {"tok": None, "model": None}


def probe_classes(p):
    return p.named_steps["logisticregression"].classes_


def format_tools(tool_ids, max_desc_words=None):
    out = []
    for tid in tool_ids:
        d = TOOL_DESC[tid]
        if max_desc_words is not None:
            d = " ".join(d.split()[:max_desc_words])
        out.append(f"- {tid}: {d}")
    return "\n".join(out)


def parse_tool(text):
    t = text.strip()
    try:
        v = json.loads(t)
        if isinstance(v, str) and v.strip() in VALID_SORTED:
            return v.strip()
        if isinstance(v, dict):
            for val in v.values():
                if isinstance(val, str) and val.strip() in VALID_SORTED:
                    return val.strip()
    except Exception:
        pass
    for tid in VALID_SORTED:
        if tid in text:
            return tid
    return None


def load_llama():
    if _LLAMA["model"] is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {LLAMA_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    if device == "cpu":
        model = model.to(device)
    _LLAMA["tok"], _LLAMA["model"] = tok, model
    print("Llama loaded.", flush=True)


# ---------------------------------------------------------------------------
# Weave scorers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Weave models
# ---------------------------------------------------------------------------

class TwoStageProbeModel(weave.Model):
    name: str = "Two-Stage-Probe-Qwen-L24"

    @weave.op
    def predict(self, query_text: str) -> dict:
        gidx = ROW_FOR_QUERY.get(query_text)
        x = X[gidx:gidx + 1]
        t0 = time.perf_counter()
        agent = str(ORCH.predict(x)[0])
        probe = TOOL_PROBES[agent]
        classes = probe_classes(probe)
        proba = probe.predict_proba(x)[0]
        top5 = [str(t) for t in classes[np.argsort(-proba)[:TOP_K]]]
        ms = (time.perf_counter() - t0) * 1000
        return {"predicted_tool": top5[0], "predicted_agent": agent,
                "top5_tools": top5, "token_count": 50, "latency_ms": ms}


class RAGModel(weave.Model):
    name: str = "RAG-MiniLM-top5"

    @weave.op
    def predict(self, query_text: str) -> dict:
        t0 = time.perf_counter()
        q = EMBEDDER.encode([query_text], normalize_embeddings=True)[0]
        sims = TOOL_EMB @ q
        top5 = [str(t) for t in np.array(TOOL_IDS)[np.argsort(-sims)[:TOP_K]]]
        ms = (time.perf_counter() - t0) * 1000
        return {"predicted_tool": top5[0],
                "predicted_agent": TOOL_TO_AGENT.get(top5[0], "unknown"),
                "top5_tools": top5, "token_count": 800, "latency_ms": ms}


class NaiveLlamaModel(weave.Model):
    name: str = "Naive-Llama-3.1-8B"

    @weave.op
    def predict(self, query_text: str) -> dict:
        import torch
        tok, model = _LLAMA["tok"], _LLAMA["model"]
        user = (f"Available tools:\n{ALL_TOOLS_BLOCK}\n\nQuery: {query_text}\n\n"
                "Return the single best tool_id as a JSON string.")
        prompt = f"{ROUTER_SYSTEM}\n\n{user}"
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=4096).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        ms = (time.perf_counter() - t0) * 1000
        pred = parse_tool(text) or "stub"
        return {"predicted_tool": pred,
                "predicted_agent": TOOL_TO_AGENT.get(pred, "unknown"),
                "top5_tools": [pred], "token_count": NAIVE_TOKENS, "latency_ms": ms}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global X, ROW_FOR_QUERY, ORCH, TOOL_PROBES, TOOL_IDS, TOOL_DESC
    global TOOL_TO_AGENT, VALID_SORTED, TOOL_EMB, EMBEDDER, ALL_TOOLS_BLOCK, NAIVE_TOKENS

    load_dotenv()
    if "WANDB_KEY" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
    elif "WANDB_API_KEY" not in os.environ:
        raise KeyError("Set WANDB_KEY or WANDB_API_KEY in your environment/.env")
    entity = os.environ.get("WANDB_ENTITY", "abrahambhatti525-santa-clara-university")
    project = os.environ.get("WANDB_PROJECT", "ToolOptim")

    # ---- data ----
    X_parts, rows = [], []
    for cache_path, query_path in DATASETS:
        Xi = np.load(cache_path)
        with open(query_path) as f:
            ri = json.load(f)
        assert Xi.shape[0] == len(ri)
        X_parts.append(Xi)
        rows.extend(ri)
    X = np.vstack(X_parts)
    y_tool = np.array([r["tool_id"] for r in rows])
    y_agent = np.array([r["agent_id"] for r in rows])
    qtext = np.array([r["query_text"] for r in rows], dtype=object)

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    agent_ids = [a["agent_id"] for a in registry["agents"]]
    TOOL_IDS = [t["tool_id"] for t in registry["tools"]]
    TOOL_DESC = {t["tool_id"]: t["description"] for t in registry["tools"]}
    TOOL_TO_AGENT = {t["tool_id"]: t["agent_id"] for t in registry["tools"]}
    VALID_SORTED = sorted(TOOL_IDS, key=len, reverse=True)
    ALL_TOOLS_BLOCK = format_tools(TOOL_IDS, max_desc_words=None)
    NAIVE_TOKENS = int(len(ALL_TOOLS_BLOCK) / 4)

    ORCH = joblib.load(ORCH_PATH)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    TOOL_PROBES = {aid: joblib.load(manifest["tool_probes"][aid]["path"])
                   for aid in agent_ids}

    # ---- reproduce test split, take small stratified sample ----
    idx = np.arange(len(y_tool))
    _, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_tool)
    te = np.array(te)
    # stratify the sample by agent so all 5 are represented
    rng = np.random.default_rng(SEED)
    per_agent = max(1, SAMPLE_N // len(agent_ids))
    pick = []
    for aid in agent_ids:
        pool = te[y_agent[te] == aid]
        pick.extend(rng.choice(pool, size=min(per_agent, len(pool)), replace=False))
    sample = np.array(pick)
    print(f"Sample for 3-way eval: {len(sample)} queries "
          f"({per_agent}/agent across {len(agent_ids)} agents).")

    ROW_FOR_QUERY = {}
    ds_rows = []
    for gidx in sample:
        gidx = int(gidx)
        qt = str(qtext[gidx])
        ROW_FOR_QUERY[qt] = gidx
        ds_rows.append({"query_text": qt, "tool_id": str(y_tool[gidx]),
                        "agent_id": str(y_agent[gidx])})

    # ---- RAG embeddings ----
    from sentence_transformers import SentenceTransformer
    EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    TOOL_EMB = EMBEDDER.encode([TOOL_DESC[t] for t in TOOL_IDS],
                               normalize_embeddings=True, show_progress_bar=False)

    # ---- Llama for real naive ----
    load_llama()

    # ---- Weave eval ----
    weave.init(f"{entity}/{project}")
    dataset = weave.Dataset(name="three_way_eval_sample_true_naive_literal", rows=ds_rows)
    weave.publish(dataset)
    evaluation = weave.Evaluation(
        name="probe_vs_rag_vs_true_naive_LITERAL",
        dataset=dataset,
        scorers=[tool_correct, tool_in_top5, agent_correct, latency_ms, tokens_used],
    )

    print("Evaluating Two-Stage probe ...", flush=True)
    asyncio.run(evaluation.evaluate(TwoStageProbeModel()))
    print("Evaluating RAG ...", flush=True)
    asyncio.run(evaluation.evaluate(RAGModel()))
    print("Evaluating REAL Naive (Llama-3.1-8B) ...", flush=True)
    asyncio.run(evaluation.evaluate(NaiveLlamaModel()))

    print(f"\nDone. 3-way eval 'probe_vs_rag_vs_true_naive_LITERAL' logged to "
          f"{entity}/{project} -> Evals tab (n={len(sample)}).")


if __name__ == "__main__":
    main()
