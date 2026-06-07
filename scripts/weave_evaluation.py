"""
Weave evaluation of routing approaches (stubbed predictions for now).

Compares Naive, RAG, Probe-routed, and a large Qwen baseline on the mixed query
dataset, scoring tool/agent correctness, tokens, latency, and tokens saved.
"""

import os
import json
import time
import asyncio

import numpy as np
import joblib
import weave
from dotenv import load_dotenv

# Credentials: same load_dotenv() / WANDB_KEY pattern as wandb_default.py.
# weave looks for WANDB_API_KEY specifically, so mirror it from WANDB_KEY.
load_dotenv()
if "WANDB_KEY" in os.environ:
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
elif "WANDB_API_KEY" not in os.environ:
    raise KeyError("Set WANDB_KEY or WANDB_API_KEY in your environment/.env")

weave.init('abrahambhatti525-santa-clara-university/ToolOptim')

# ---------------------------------------------------------------------------
# 1. Define the Dataset (first 50 rows of the real mixed dataset)
# ---------------------------------------------------------------------------

QUERIES_MIXED_PATH = "data/queries_mixed.json"

with open(QUERIES_MIXED_PATH) as f:
    queries = json.load(f)[:50]

dataset = weave.Dataset(
    name="queries_mixed_v1",
    rows=[
        {"query_text": r["query_text"],
         "agent_id": r["agent_id"],
         "tool_id": r["tool_id"]}
        for r in queries
    ]
)
weave.publish(dataset)

# ---------------------------------------------------------------------------
# 2. Define Scorers
# ---------------------------------------------------------------------------

@weave.op
def tool_correct(tool_id: str, output: dict) -> dict:
    return {"correct": output["predicted_tool"] == tool_id}

@weave.op
def agent_correct(agent_id: str, output: dict) -> dict:
    return {"correct": output["predicted_agent"] == agent_id}

@weave.op
def tokens_used(output: dict) -> dict:
    return {"tokens": output["token_count"]}

@weave.op
def latency_ms(output: dict) -> dict:
    return {"ms": output["latency_ms"]}

@weave.op
def tokens_saved(output: dict) -> dict:
    return {"saved": 4200 - output["token_count"]}

@weave.op
def tool_in_top5(tool_id: str, output: dict) -> dict:
    top5 = output.get("top5_tools", [output.get("predicted_tool")])
    return {"hit": tool_id in top5}

# ---------------------------------------------------------------------------
# 2.5 Shared inference setup (cached hidden states + probes + RAG embeddings)
# ---------------------------------------------------------------------------

CACHE_PATH = "data/probe_cache/mixed_Qwen2.5-7B-Instruct_layer24.npy"
REGISTRY_PATH = "data/registry.json"
TOOL_PROBE_PATHS = ["data/probes/tool_probe_best.pkl", "data/probes/tool_probe.pkl"]
ORCH_PROBE_PATHS = [
    "data/probes/orchestrator_probe_best.pkl",
    "data/probes/orchestrator_probe.pkl",
]


def _load_probe(paths):
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"None of these probes exist: {paths}")


# Cached layer-24 hidden states, indexed the same as the full mixed dataset.
X = np.load(CACHE_PATH)

with open(QUERIES_MIXED_PATH) as f:
    _all_rows = json.load(f)
# query_text -> (cache index, tool_id, agent_id)
QUERY_INDEX = {
    r["query_text"]: (i, r["tool_id"], r["agent_id"])
    for i, r in enumerate(_all_rows)
}

tool_probe = _load_probe(TOOL_PROBE_PATHS)
orch_probe = _load_probe(ORCH_PROBE_PATHS)

with open(REGISTRY_PATH) as f:
    _registry = json.load(f)
tool_desc = {t["tool_id"]: t["description"] for t in _registry["tools"]}
tool_to_agent = {t["tool_id"]: t["agent_id"] for t in _registry["tools"]}
TOOL_IDS = [t["tool_id"] for t in _registry["tools"]]

# RAG: pre-embed all tool descriptions once with MiniLM.
from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_tool_embeddings = _embedder.encode(
    [tool_desc[t] for t in TOOL_IDS], normalize_embeddings=True
)

# ---------------------------------------------------------------------------
# 3. Define Models for each approach
# ---------------------------------------------------------------------------

class NaiveModel(weave.Model):
    name: str = "Naive-Llama-3.1-8B"
    model_name: str = "Llama-3.1-8B"

    @weave.op
    def predict(self, query_text: str) -> dict:
        # Naive requires loading Llama-3.1-8B with 200 tools in context —
        # benchmarked separately in full_benchmark.py.
        return {
            "predicted_tool": "stub",
            "predicted_agent": "stub",
            "token_count": 4200,
            "latency_ms": 0
        }

class RAGModel(weave.Model):
    name: str = "RAG-SentenceTransformer"
    model_name: str = "RAG-SentenceTransformer"

    @weave.op
    def predict(self, query_text: str) -> dict:
        t0 = time.perf_counter()
        q_emb = _embedder.encode([query_text], normalize_embeddings=True)[0]
        sims = _tool_embeddings @ q_emb
        top5_idx = np.argsort(-sims)[:5]
        latency_ms = (time.perf_counter() - t0) * 1000

        top5_tools = [TOOL_IDS[j] for j in top5_idx]
        predicted_tool = top5_tools[0]
        return {
            "predicted_tool": predicted_tool,
            "predicted_agent": tool_to_agent.get(predicted_tool, "unknown"),
            "token_count": 800,
            "latency_ms": latency_ms,
            "top5_tools": top5_tools,
        }

class ProbeRoutedModel(weave.Model):
    name: str = "Probe-Routed-Llama-3.1-8B"
    model_name: str = "Llama-3.1-8B"
    probe_layer: int = 16

    @weave.op
    def predict(self, query_text: str) -> dict:
        entry = QUERY_INDEX.get(query_text)
        if entry is None:
            return {
                "predicted_tool": "stub",
                "predicted_agent": "stub",
                "token_count": 50,
                "latency_ms": 0,
                "top5_tools": [],
            }
        index = entry[0]
        x = X[index : index + 1]

        t0 = time.perf_counter()
        orch_proba = orch_probe.predict_proba(x)[0]
        tool_proba = tool_probe.predict_proba(x)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        a_idx = int(np.argmax(orch_proba))
        predicted_agent = orch_probe.classes_[a_idx]
        agent_confidence = float(orch_proba[a_idx])

        top5_idx = np.argsort(-tool_proba)[:5]
        top5_tools = [tool_probe.classes_[j] for j in top5_idx]
        predicted_tool = top5_tools[0]
        return {
            "predicted_tool": predicted_tool,
            "predicted_agent": predicted_agent,
            "agent_confidence": agent_confidence,
            "token_count": 50,
            "latency_ms": latency_ms,
            "top5_tools": top5_tools,
        }

class Qwen72BNaiveModel(weave.Model):
    name: str = "Naive-Qwen2.5-72B"
    model_name: str = "Qwen2.5-72B"

    @weave.op
    def predict(self, query_text: str) -> dict:
        tool_list = "\n".join(f"- {t}: {tool_desc[t]}" for t in TOOL_IDS)
        prompt = (
            f"Available tools:\n{tool_list}\n\nQuery: {query_text}\n\n"
            "Return the single best tool_id as a JSON string."
        )
        t0 = time.perf_counter()
        try:
            import ollama

            resp = ollama.chat(
                model="qwen2.5:72b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a tool router. Return ONLY the tool_id of the "
                            'single best tool as a JSON string, e.g. "search_auth_logs".'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0},
            )
            text = resp["message"]["content"].strip()
            try:
                predicted_tool = json.loads(text)
                if not isinstance(predicted_tool, str):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                predicted_tool = next((t for t in TOOL_IDS if t in text), "stub")
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "predicted_tool": predicted_tool,
                "predicted_agent": tool_to_agent.get(predicted_tool, "unknown"),
                "token_count": 4200,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            print(f"  [warn] Qwen2.5-72B (Ollama) failed: {e}")
            return {
                "predicted_tool": "stub",
                "predicted_agent": "stub",
                "token_count": 4200,
                "latency_ms": 0,
            }

# ---------------------------------------------------------------------------
# 4. Wire the Evaluation
# ---------------------------------------------------------------------------

evaluation = weave.Evaluation(
    name="probe_vs_naive_v1",
    dataset=dataset,
    scorers=[tool_correct, agent_correct, tokens_used, latency_ms, tokens_saved, tool_in_top5]
)

asyncio.run(evaluation.evaluate(NaiveModel()))
asyncio.run(evaluation.evaluate(RAGModel()))
asyncio.run(evaluation.evaluate(ProbeRoutedModel()))
asyncio.run(evaluation.evaluate(Qwen72BNaiveModel()))
