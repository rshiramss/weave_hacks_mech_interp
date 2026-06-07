"""
Benchmark four tool-selection approaches on a 200-query sample of
data/queries_mixed.json against data/registry.json:

  1. Naive      - Llama-3.1-8B with all 200 tool descriptions in context
  2. RAG        - all-MiniLM-L6-v2 retrieval (top-5) + Llama-3.1-8B executor
  3. Probe      - Qwen2.5-7B layer-24 LogReg probe (top-5) + Llama-3.1-8B executor
  4. Frontier   - qwen2.5:72b via Ollama with all tool descriptions

Reports top-1 accuracy, recall@5, avg ms/query, and est. tokens/query, prints a
summary table, and logs a single W&B run.

Run from the project root:
    python scripts/full_benchmark.py
"""

import os
import gc
import json
import time
import random
import asyncio
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import joblib
import wandb
import weave
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REGISTRY_PATH = "data/registry.json"
QUERIES_PATH = "data/queries_mixed.json"
PROBE_PATH = "data/probes/tool_probe_best.pkl" if os.path.exists("data/probes/tool_probe_best.pkl") else "data/probes/tool_probe.pkl"
CACHE_PATH = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy"

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
FRONTIER_MODEL = "qwen2.5:72b"

N_SAMPLES = 200
SEED = 42
TOPK = 5

ROUTER_SYSTEM = (
    "You are a tool router for a SOC. Given a user query and a list of available "
    "tools, return ONLY the tool_id of the single best tool as a JSON string, "
    'e.g. "search_auth_logs". No explanation, no markdown.'
)

# ---------------------------------------------------------------------------
# W&B credentials (same load_dotenv() / WANDB_API_KEY pattern as other scripts)
# ---------------------------------------------------------------------------

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
# Data
# ---------------------------------------------------------------------------

def load_registry():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    tool_ids = [t["tool_id"] for t in registry["tools"]]
    tool_desc = {t["tool_id"]: t["description"] for t in registry["tools"]}
    return tool_ids, tool_desc


def load_sample():
    """Return (sample_indices, queries, tool_labels, agent_labels) for a sample."""
    with open(QUERIES_PATH) as f:
        raw = json.load(f)
    rng = random.Random(SEED)
    n = min(N_SAMPLES, len(raw))
    idx = rng.sample(range(len(raw)), n)
    queries = [raw[i]["query_text"] for i in idx]
    labels = [raw[i]["tool_id"] for i in idx]
    agents = [raw[i]["agent_id"] for i in idx]
    return idx, queries, labels, agents


def format_tools(tool_ids, tool_desc, max_desc_words=None):
    rows = []
    for tid in tool_ids:
        desc = tool_desc[tid]
        if max_desc_words is not None:
            desc = " ".join(desc.split()[:max_desc_words])
        rows.append(f"- {tid}: {desc}")
    return "\n".join(rows)


def parse_tool(text, valid_sorted):
    """Extract a valid tool_id from raw model output."""
    t = text.strip()
    try:
        v = json.loads(t)
        if isinstance(v, str):
            v = v.strip()
            if v in valid_sorted:
                return v
        if isinstance(v, dict):
            for val in v.values():
                if isinstance(val, str):
                    val = val.strip()
                if isinstance(val, str) and val in valid_sorted:
                    return val
    except Exception:
        pass
    for tid in valid_sorted:
        if tid in text:
            return tid
    return None


# ---------------------------------------------------------------------------
# Llama-3.1-8B executor (loaded once, reused by Naive / RAG / Probe)
# ---------------------------------------------------------------------------

class LlamaExecutor:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.raw_output_printed = False
        self.parse_success = 0
        self.parse_failed = 0
        self.last_raw = ""

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {LLAMA_MODEL} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()
        if device == "cpu":
            self.model = self.model.to(device)

    def pick(self, query, tool_ids, tool_desc, valid_sorted, max_desc_words=None):
        import torch

        user = (
            f"Available tools:\n{format_tools(tool_ids, tool_desc, max_desc_words)}\n\n"
            f"Query: {query}\n\nReturn the single best tool_id as a JSON string."
        )
        prompt = f"{ROUTER_SYSTEM}\n\n{user}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        self.last_raw = text
        if not self.raw_output_printed:
            print("\n[Llama raw output for first query]")
            print(text)
            print("[end raw output]\n")
            self.raw_output_printed = True
        pred = parse_tool(text, valid_sorted)
        if pred is None:
            self.parse_failed += 1
        else:
            self.parse_success += 1
        return pred

    def print_parse_stats(self):
        print(
            f"Llama parse stats: {self.parse_success} succeeded, "
            f"{self.parse_failed} failed"
        )

    def unload(self):
        import torch

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Approaches
# ---------------------------------------------------------------------------

def progress(approach, i, n):
    if (i + 1) % 10 == 0 or (i + 1) == n:
        print(f"  [{approach}] {i + 1}/{n}")


def run_naive(executor, queries, labels, tool_ids, tool_desc, valid_sorted, per_query):
    all_desc = format_tools(tool_ids, tool_desc, max_desc_words=20)
    tokens_per_q = len(all_desc) / 4
    correct = 0
    times = []
    for i, (q, gt) in enumerate(zip(queries, labels)):
        t0 = time.perf_counter()
        try:
            pred = executor.pick(q, tool_ids, tool_desc, valid_sorted, max_desc_words=20)
            if i == 0:
                print(
                    f"  [naive] query 0 raw output (first 200 chars): "
                    f"{executor.last_raw[:200]!r}"
                )
        except Exception as e:
            if i == 0:
                import traceback
                traceback.print_exc()
            print(f"  [naive] query {i} failed: {type(e).__name__}: {e}")
            pred = None
        ms_q = (time.perf_counter() - t0) * 1000
        times.append(ms_q)
        per_query[q] = {
            "predicted_tool": pred if pred is not None else "stub",
            "top5_tools": [],
            "token_count": tokens_per_q,
            "latency_ms": ms_q,
        }
        if pred == gt:
            correct += 1
        progress("Naive", i, len(queries))
    return {
        "approach": "Naive (Llama-8B)",
        "top1": correct / len(queries),
        "recall5": None,
        "ms": float(np.mean(times)),
        "tokens": tokens_per_q,
    }


def run_rag(executor, queries, labels, tool_ids, tool_desc, valid_sorted, per_query):
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBED_MODEL)
    desc_list = [tool_desc[t] for t in tool_ids]
    tool_emb = embedder.encode(desc_list, normalize_embeddings=True)

    correct = 0
    recall_hits = 0
    times = []
    tokens = []
    for i, (q, gt) in enumerate(zip(queries, labels)):
        t0 = time.perf_counter()
        try:
            q_emb = embedder.encode([q], normalize_embeddings=True)[0]
            sims = tool_emb @ q_emb
            top5 = [tool_ids[j] for j in np.argsort(-sims)[:TOPK]]
            pred = executor.pick(q, top5, tool_desc, valid_sorted)
        except Exception as e:
            print(f"  [rag] query {i} failed: {type(e).__name__}: {e}")
            top5, pred = [], None
        ms_q = (time.perf_counter() - t0) * 1000
        times.append(ms_q)
        tok = len(format_tools(top5, tool_desc)) / 4 if top5 else 0
        tokens.append(tok)
        per_query[q] = {
            "predicted_tool": pred if pred is not None else "stub",
            "top5_tools": list(top5),
            "token_count": tok,
            "latency_ms": ms_q,
        }
        if gt in top5:
            recall_hits += 1
        if pred == gt:
            correct += 1
        progress("RAG", i, len(queries))
    return {
        "approach": "RAG (MiniLM+8B)",
        "top1": correct / len(queries),
        "recall5": recall_hits / len(queries),
        "ms": float(np.mean(times)),
        "tokens": float(np.mean(tokens)),
    }


def run_probe(executor, sample_idx, queries, labels, tool_ids, tool_desc, valid_sorted, per_query):
    probe = joblib.load(PROBE_PATH)
    X = np.load(CACHE_PATH)
    print(f"  [probe] first sampled index: {sample_idx[0]}, X shape: {X.shape}")
    X_sample = X[sample_idx]

    # ms/query = probe inference only (extraction is a one-time offline cost).
    t0 = time.perf_counter()
    proba = probe.predict_proba(X_sample)
    probe_ms = (time.perf_counter() - t0) * 1000 / len(sample_idx)

    classes = probe.classes_
    top5_idx = np.argsort(-proba, axis=1)[:, :TOPK]

    correct = 0
    recall_hits = 0
    tokens = []
    for i, (q, gt) in enumerate(zip(queries, labels)):
        top5 = [classes[j] for j in top5_idx[i]]
        if gt in top5:
            recall_hits += 1
        try:
            pred = executor.pick(q, top5, tool_desc, valid_sorted)
        except Exception as e:
            print(f"  [probe] query {i} failed: {type(e).__name__}: {e}")
            pred = None
        tok = len(format_tools(top5, tool_desc)) / 4
        tokens.append(tok)
        per_query[q] = {
            "predicted_tool": pred if pred is not None else "stub",
            "top5_tools": list(top5),
            "token_count": tok,
            "latency_ms": probe_ms,
        }
        if pred == gt:
            correct += 1
        progress("Probe", i, len(queries))
    return {
        "approach": "Probe (Qwen7B+8B)",
        "top1": correct / len(queries),
        "recall5": recall_hits / len(queries),
        "ms": probe_ms,
        "tokens": float(np.mean(tokens)),
    }


def run_frontier(queries, labels, tool_ids, tool_desc, valid_sorted, per_query):
    try:
        import ollama
    except ImportError:
        print("  [frontier] ollama package not installed; skipping.")
        return None

    all_desc = format_tools(tool_ids, tool_desc)
    tokens_per_q = len(all_desc) / 4
    correct = 0
    times = []
    for i, (q, gt) in enumerate(zip(queries, labels)):
        user = (
            f"Available tools:\n{all_desc}\n\nQuery: {q}\n\n"
            "Return the single best tool_id as a JSON string."
        )
        t0 = time.perf_counter()
        try:
            resp = ollama.chat(
                model=FRONTIER_MODEL,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0},
            )
            pred = parse_tool(resp["message"]["content"], valid_sorted)
        except Exception as e:
            if i == 0:
                print(f"  [frontier] Ollama unavailable ({e}); skipping approach.")
                return None
            print(f"  [frontier] query {i} failed: {type(e).__name__}: {e}")
            pred = None
        ms_q = (time.perf_counter() - t0) * 1000
        times.append(ms_q)
        per_query[q] = {
            "predicted_tool": pred if pred is not None else "stub",
            "top5_tools": [],
            "token_count": tokens_per_q,
            "latency_ms": ms_q,
        }
        if pred == gt:
            correct += 1
        progress("Frontier", i, len(queries))
    return {
        "approach": "Frontier (72B)",
        "top1": correct / len(queries),
        "recall5": None,
        "ms": float(np.mean(times)),
        "tokens": tokens_per_q,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt(v, spec):
    return "—" if v is None else format(v, spec)


def print_table(results):
    header = (
        f"{'Approach':<18}{'Top-1 Acc':>11}{'Recall@5':>11}"
        f"{'ms/query':>11}{'Tokens/query':>14}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['approach']:<18}{fmt(r['top1'], '.4f'):>11}"
            f"{fmt(r['recall5'], '.4f'):>11}{fmt(r['ms'], '.1f'):>11}"
            f"{fmt(r['tokens'], '.0f'):>14}"
        )


# approach name -> (job_type, clean run name) for the per-approach W&B runs.
WANDB_RUN_META = {
    "Naive (Llama-8B)": ("naive", "naive-llama-8b"),
    "RAG (MiniLM+8B)": ("rag", "rag-minilm-llama-8b"),
    "Probe (Qwen7B+8B)": ("probe", "probe-qwen7b-layer24"),
    "Frontier (72B)": ("frontier", "frontier-qwen-72b"),
}


def log_to_wandb(results, entity, project):
    """Log one W&B run per approach, all in group 'full-benchmark' so they line
    up side by side in the comparison table."""
    for r in results:
        job_type, name = WANDB_RUN_META.get(r["approach"], ("unknown", r["approach"]))
        wandb.init(
            entity=entity,
            project=project,
            name=name,
            group="full-benchmark",
            job_type=job_type,
            tags=["full_benchmark", "comparison"],
            reinit=True,
        )
        metrics = {
            "top1_acc": r["top1"],
            "ms_per_query": r["ms"],
            "tokens_per_query": r["tokens"],
        }
        if r.get("recall5") is not None:
            metrics["recall@5"] = r["recall5"]
        wandb.log(metrics)
        wandb.summary.update(metrics)
        wandb.finish()
        print(f"Logged W&B run '{name}' (group=full-benchmark, job_type={job_type}).")


# ---------------------------------------------------------------------------
# Weave evaluation logging (replays already-computed per-query results; no
# re-running of inference). Same pattern as scripts/weave_evaluation.py.
# ---------------------------------------------------------------------------

# Populated by run_weave_eval() before the Weave models are evaluated:
#   APPROACH_RESULTS[approach_name][query_text] = result_dict
APPROACH_RESULTS = {}


def _lookup_result(approach: str, query_text: str) -> dict:
    res = APPROACH_RESULTS.get(approach, {}).get(query_text)
    if res is None:
        return {
            "predicted_tool": "stub",
            "top5_tools": [],
            "token_count": 0,
            "latency_ms": 0,
        }
    return res


@weave.op
def tool_correct(tool_id: str, output: dict) -> dict:
    return {"correct": output["predicted_tool"] == tool_id}


@weave.op
def tool_in_top5(tool_id: str, output: dict) -> dict:
    return {"hit": tool_id in output.get("top5_tools", [])}


@weave.op
def latency_ms(output: dict) -> dict:
    return {"ms": output["latency_ms"]}


@weave.op
def tokens_used(output: dict) -> dict:
    return {"tokens": output["token_count"]}


class NaiveLlamaModel(weave.Model):
    name: str = "Naive-Llama-3.1-8B"

    @weave.op
    def predict(self, query_text: str) -> dict:
        return _lookup_result("Naive (Llama-8B)", query_text)


class RAGModel(weave.Model):
    name: str = "RAG-MiniLM-Llama-3.1-8B"

    @weave.op
    def predict(self, query_text: str) -> dict:
        return _lookup_result("RAG (MiniLM+8B)", query_text)


class ProbeRoutedModel(weave.Model):
    name: str = "Probe-Routed-Qwen7B-Llama-3.1-8B"

    @weave.op
    def predict(self, query_text: str) -> dict:
        return _lookup_result("Probe (Qwen7B+8B)", query_text)


class FrontierQwen72BModel(weave.Model):
    name: str = "Naive-Qwen2.5-72B"

    @weave.op
    def predict(self, query_text: str) -> dict:
        return _lookup_result("Frontier (72B)", query_text)


def run_weave_eval(queries, labels, agents, approach_results, frontier_present):
    """Replay benchmark results through a Weave Evaluation (no re-inference)."""
    global APPROACH_RESULTS
    APPROACH_RESULTS = approach_results
    try:
        load_dotenv()
        if "WANDB_KEY" in os.environ:
            os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
        elif "WANDB_API_KEY" not in os.environ:
            raise KeyError("Set WANDB_KEY or WANDB_API_KEY in your environment/.env")

        weave.init('abrahambhatti525-santa-clara-university/ToolOptim')

        dataset = weave.Dataset(
            name="full_benchmark_sample",
            rows=[
                {"query_text": q, "tool_id": t, "agent_id": a}
                for q, t, a in zip(queries, labels, agents)
            ],
        )
        weave.publish(dataset)

        evaluation = weave.Evaluation(
            name="full_benchmark_eval",
            dataset=dataset,
            scorers=[tool_correct, tool_in_top5, latency_ms, tokens_used],
        )

        asyncio.run(evaluation.evaluate(NaiveLlamaModel()))
        asyncio.run(evaluation.evaluate(RAGModel()))
        asyncio.run(evaluation.evaluate(ProbeRoutedModel()))
        if frontier_present:
            asyncio.run(evaluation.evaluate(FrontierQwen72BModel()))

        print("Weave evaluation logged.")
    except Exception as e:
        print(f"  [warn] Weave evaluation logging failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tool_ids, tool_desc = load_registry()
    valid_sorted = sorted(tool_ids, key=len, reverse=True)
    sample_idx, queries, labels, agents = load_sample()
    print(f"Benchmarking {len(queries)} queries over {len(tool_ids)} tools.")

    key, entity, project = load_wandb_config()

    results = []
    approach_results = {}  # approach_name -> {query_text -> result_dict}
    executor = LlamaExecutor()
    executor.load()

    print("\n=== Approach 1: Naive ===")
    naive_pq = {}
    results.append(
        run_naive(executor, queries, labels, tool_ids, tool_desc, valid_sorted, naive_pq)
    )
    approach_results["Naive (Llama-8B)"] = naive_pq

    print("\n=== Approach 2: RAG ===")
    rag_pq = {}
    results.append(
        run_rag(executor, queries, labels, tool_ids, tool_desc, valid_sorted, rag_pq)
    )
    approach_results["RAG (MiniLM+8B)"] = rag_pq

    print("\n=== Approach 3: Probe ===")
    probe_pq = {}
    results.append(
        run_probe(
            executor, sample_idx, queries, labels, tool_ids, tool_desc, valid_sorted, probe_pq
        )
    )
    approach_results["Probe (Qwen7B+8B)"] = probe_pq

    executor.print_parse_stats()

    # Free Llama VRAM before the Ollama-served frontier model.
    executor.unload()

    print("\n=== Approach 4: Frontier ===")
    frontier_pq = {}
    frontier = run_frontier(queries, labels, tool_ids, tool_desc, valid_sorted, frontier_pq)
    frontier_present = frontier is not None
    if frontier is not None:
        results.append(frontier)
        approach_results["Frontier (72B)"] = frontier_pq

    print_table(results)

    run_weave_eval(queries, labels, agents, approach_results, frontier_present)

    try:
        wandb.login(key=key)
        log_to_wandb(results, entity, project)
    except Exception as e:
        print(f"  [warn] W&B logging failed: {e}")


if __name__ == "__main__":
    main()
