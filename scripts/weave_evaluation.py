"""
Weave evaluation of routing approaches (stubbed predictions for now).

Compares Naive, RAG, Probe-routed, and a large Qwen baseline on the mixed query
dataset, scoring tool/agent correctness, tokens, latency, and tokens saved.
"""

import os
import json
import asyncio

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

# ---------------------------------------------------------------------------
# 3. Define Model stubs for each approach
# ---------------------------------------------------------------------------

class NaiveModel(weave.Model):
    name: str = "Naive-Llama-3.1-8B"
    model_name: str = "Llama-3.1-8B"

    @weave.op
    def predict(self, query_text: str) -> dict:
        # stub for now, wire real inference later
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
        return {
            "predicted_tool": "stub",
            "predicted_agent": "stub",
            "token_count": 800,
            "latency_ms": 0
        }

class ProbeRoutedModel(weave.Model):
    name: str = "Probe-Routed-Llama-3.1-8B"
    model_name: str = "Llama-3.1-8B"
    probe_layer: int = 16

    @weave.op
    def predict(self, query_text: str) -> dict:
        return {
            "predicted_tool": "stub",
            "predicted_agent": "stub",
            "token_count": 800,
            "latency_ms": 0
        }

class Qwen72BNaiveModel(weave.Model):
    name: str = "Naive-Qwen2.5-72B"
    model_name: str = "Qwen2.5-72B"

    @weave.op
    def predict(self, query_text: str) -> dict:
        return {
            "predicted_tool": "stub",
            "predicted_agent": "stub",
            "token_count": 4200,
            "latency_ms": 0
        }

# ---------------------------------------------------------------------------
# 4. Wire the Evaluation
# ---------------------------------------------------------------------------

evaluation = weave.Evaluation(
    name="probe_vs_naive_v1",
    dataset=dataset,
    scorers=[tool_correct, agent_correct, tokens_used, latency_ms, tokens_saved]
)

asyncio.run(evaluation.evaluate(NaiveModel()))
asyncio.run(evaluation.evaluate(RAGModel()))
asyncio.run(evaluation.evaluate(ProbeRoutedModel()))
asyncio.run(evaluation.evaluate(Qwen72BNaiveModel()))
