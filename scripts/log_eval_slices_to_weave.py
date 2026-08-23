"""Log two held-out eval slices to W&B / Weave Evals (6 eval nodes total).

Two slices, each evaluated with 3 models => 6 eval model-runs ("eval nodes"):

  Slice 1 "hard_no_cue_..."  -- held-out rows with NO lexical cue (tool phrase,
      tool tokens, or first description words). Adversarial; labels can be noisy.
  Slice 2 "probe_strength_slice_..." -- held-out rows with no exact tool phrase
      where the two-stage probe is top-1 correct AND RAG top-1 is wrong (naive is
      NOT used in selection). A curated diagnostic slice, not an unbiased
      benchmark.

Both slices are reproduced deterministically (seed 42) so they match the local
runs. Models: Two-stage probe, RAG (MiniLM top-5), true naive Llama-3.1-8B with
all 200 full tool descriptions.

Run from the project root:
    python scripts/log_eval_slices_to_weave.py
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

# globals used by weave models
X = None
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


def format_tools(tool_ids):
    return "\n".join(f"- {tid}: {TOOL_DESC[tid]}" for tid in tool_ids)


def parse_tool(text):
    text = text.strip()
    try:
        v = json.loads(text)
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


def probe_classes(p):
    return p.named_steps["logisticregression"].classes_


def has_obvious_cue(text, tool_id):
    text = text.lower()
    phrase = tool_id.replace("_", " ")
    tool_tokens = [t for t in tool_id.split("_") if len(t) > 3]
    desc_tokens = [
        w.strip(".,:;()[]").lower()
        for w in TOOL_DESC[tool_id].split()
        if len(w.strip(".,:;()[]")) > 5
    ]
    return (
        phrase in text
        or any(t in text for t in tool_tokens)
        or any(t in text for t in desc_tokens[:4])
    )


def exact_tool_phrase(text, tool_id):
    return tool_id.replace("_", " ") in text.lower()


def load_llama():
    if _LLAMA["model"] is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
# Scorers
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
# Models (all share the same predict signature; cache_idx comes from the row)
# ---------------------------------------------------------------------------

class TwoStageProbeModel(weave.Model):
    name: str = "Two-Stage-Probe-Qwen-L24"

    @weave.op
    def predict(self, query_text: str, cache_idx: int) -> dict:
        x = X[cache_idx:cache_idx + 1]
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
    def predict(self, query_text: str, cache_idx: int) -> dict:
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
    def predict(self, query_text: str, cache_idx: int) -> dict:
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
        ms = (time.perf_counter() - t0) * 1000
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        pred = parse_tool(text) or "stub"
        return {"predicted_tool": pred,
                "predicted_agent": TOOL_TO_AGENT.get(pred, "unknown"),
                "top5_tools": [pred], "token_count": NAIVE_TOKENS, "latency_ms": ms}


# ---------------------------------------------------------------------------
# Slice builders (reproduce the local runs deterministically)
# ---------------------------------------------------------------------------

def build_no_cue_slice(test_idx, y_tool, y_agent, qtext, agent_ids):
    hard = [i for i in test_idx if not has_obvious_cue(str(qtext[i]), str(y_tool[i]))]
    rng = np.random.default_rng(SEED)
    sample = []
    for aid in agent_ids:
        pool = np.array([i for i in hard if y_agent[i] == aid])
        take = min(PER_AGENT, len(pool))
        sample.extend(rng.choice(pool, size=take, replace=False))
    return [int(i) for i in sample], len(hard)


def build_probe_strength_slice(test_idx, y_tool, y_agent, qtext, agent_ids):
    # probe + RAG predictions across the test split
    q_emb = EMBEDDER.encode(list(qtext[test_idx]), normalize_embeddings=True,
                            batch_size=256, show_progress_bar=False)
    rag_order = np.argsort(-(q_emb @ TOOL_EMB.T), axis=1)
    tool_arr = np.array(TOOL_IDS)

    cand = {aid: [] for aid in agent_ids}
    for pos, i in enumerate(test_idx):
        gt = str(y_tool[i])
        if exact_tool_phrase(str(qtext[i]), gt):
            continue
        agent = str(ORCH.predict(X[i:i + 1])[0])
        proba = TOOL_PROBES[agent].predict_proba(X[i:i + 1])[0]
        ptop1 = str(probe_classes(TOOL_PROBES[agent])[np.argmax(proba)])
        rtop1 = str(tool_arr[rag_order[pos, 0]])
        if ptop1 == gt and rtop1 != gt:
            cand[str(y_agent[i])].append(int(i))

    rng = np.random.default_rng(SEED)
    sample = []
    for aid in agent_ids:
        pool = cand[aid]
        take = min(PER_AGENT, len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False)
        sample.extend(pool[int(j)] for j in chosen)
    total = sum(len(cand[a]) for a in agent_ids)
    return sample, total


def run_eval(slice_name, sample_idx, y_tool, y_agent, qtext):
    rows = [{"query_text": str(qtext[i]), "cache_idx": int(i),
             "tool_id": str(y_tool[i]), "agent_id": str(y_agent[i])}
            for i in sample_idx]
    dataset = weave.Dataset(name=f"{slice_name}_dataset", rows=rows)
    weave.publish(dataset)
    evaluation = weave.Evaluation(
        name=slice_name,
        dataset=dataset,
        scorers=[tool_correct, tool_in_top5, agent_correct, latency_ms, tokens_used],
    )
    print(f"\n=== Eval '{slice_name}' (n={len(rows)}) ===", flush=True)
    print("  evaluating Two-Stage probe ...", flush=True)
    asyncio.run(evaluation.evaluate(TwoStageProbeModel()))
    print("  evaluating RAG ...", flush=True)
    asyncio.run(evaluation.evaluate(RAGModel()))
    print("  evaluating true Naive Llama ...", flush=True)
    asyncio.run(evaluation.evaluate(NaiveLlamaModel()))


def main():
    global X, ORCH, TOOL_PROBES, TOOL_IDS, TOOL_DESC, TOOL_TO_AGENT, VALID_SORTED
    global TOOL_EMB, EMBEDDER, ALL_TOOLS_BLOCK, NAIVE_TOKENS

    load_dotenv()
    if "WANDB_KEY" in os.environ:
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
    elif "WANDB_API_KEY" not in os.environ:
        raise KeyError("Set WANDB_KEY or WANDB_API_KEY in your environment/.env")
    entity = os.environ.get("WANDB_ENTITY", "abrahambhatti525-santa-clara-university")
    project = os.environ.get("WANDB_PROJECT", "ToolOptim")

    # data
    X_parts, rows = [], []
    for _, cache_path, query_path in DATASETS:
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
    ALL_TOOLS_BLOCK = format_tools(TOOL_IDS)
    NAIVE_TOKENS = int(len(ALL_TOOLS_BLOCK) / 4)

    ORCH = joblib.load(ORCH_PATH)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    TOOL_PROBES = {aid: joblib.load(manifest["tool_probes"][aid]["path"])
                   for aid in agent_ids}

    from sentence_transformers import SentenceTransformer
    EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    TOOL_EMB = EMBEDDER.encode([TOOL_DESC[t] for t in TOOL_IDS],
                               normalize_embeddings=True, show_progress_bar=False)

    idx = np.arange(len(rows))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_tool)
    test_idx = np.array(test_idx)

    print("Building slices...")
    no_cue, n_hard = build_no_cue_slice(test_idx, y_tool, y_agent, qtext, agent_ids)
    probe_slice, n_cand = build_probe_strength_slice(test_idx, y_tool, y_agent, qtext, agent_ids)
    print(f"  hard_no_cue: pool={n_hard}, sample={len(no_cue)}")
    print(f"  probe_strength_slice: pool={n_cand}, sample={len(probe_slice)}")

    load_llama()
    weave.init(f"{entity}/{project}")

    run_eval("probe_strength_slice_vs_rag_vs_true_naive", probe_slice,
             y_tool, y_agent, qtext)
    run_eval("hard_no_cue_vs_rag_vs_true_naive", no_cue,
             y_tool, y_agent, qtext)

    print("\nDone. 6 eval nodes logged (2 evaluations x 3 models) to "
          f"{entity}/{project} -> Evals tab.")


if __name__ == "__main__":
    main()
