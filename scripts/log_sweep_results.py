"""
Log fake layer-sweep results to Weights & Biases.

Mirrors the structure the real sweep (scripts/probe_ab_test.py:run_layer_sweep)
will produce: one run per (model, layer), grouped under "layer_sweep_test".

Uses the same load_dotenv() / os.environ['WANDB_API_KEY'] credential pattern and
the same W&B project as wandb_default.py. We use wandb (not weave) here because
we're logging experiment metrics, not LLM traces.
"""

import os

import wandb
from dotenv import load_dotenv

load_dotenv()

WANDB_ENTITY = "abrahambhatti525-santa-clara-university"
WANDB_PROJECT = "ToolOptim"
GROUP = "layer_sweep_test"

FAKE_RESULTS = [
    {"model": "Llama-3.1-8B-Instruct", "layer": 0,  "tool_recall5": 0.12, "agent_recall2": 0.71},
    {"model": "Llama-3.1-8B-Instruct", "layer": 4,  "tool_recall5": 0.34, "agent_recall2": 0.81},
    {"model": "Llama-3.1-8B-Instruct", "layer": 8,  "tool_recall5": 0.51, "agent_recall2": 0.88},
    {"model": "Llama-3.1-8B-Instruct", "layer": 12, "tool_recall5": 0.63, "agent_recall2": 0.92},
    {"model": "Llama-3.1-8B-Instruct", "layer": 16, "tool_recall5": 0.67, "agent_recall2": 0.93},
    {"model": "Llama-3.1-8B-Instruct", "layer": 20, "tool_recall5": 0.61, "agent_recall2": 0.91},
    {"model": "Llama-3.1-8B-Instruct", "layer": 24, "tool_recall5": 0.58, "agent_recall2": 0.90},
    {"model": "Llama-3.1-8B-Instruct", "layer": 28, "tool_recall5": 0.55, "agent_recall2": 0.89},
    {"model": "Qwen2.5-7B-Instruct",   "layer": 0,  "tool_recall5": 0.10, "agent_recall2": 0.68},
    {"model": "Qwen2.5-7B-Instruct",   "layer": 4,  "tool_recall5": 0.29, "agent_recall2": 0.78},
    {"model": "Qwen2.5-7B-Instruct",   "layer": 8,  "tool_recall5": 0.44, "agent_recall2": 0.85},
    {"model": "Qwen2.5-7B-Instruct",   "layer": 12, "tool_recall5": 0.59, "agent_recall2": 0.91},
    {"model": "Qwen2.5-7B-Instruct",   "layer": 16, "tool_recall5": 0.64, "agent_recall2": 0.92},
    {"model": "Qwen2.5-7B-Instruct",   "layer": 20, "tool_recall5": 0.60, "agent_recall2": 0.90},
    {"model": "Qwen2.5-7B-Instruct",   "layer": 24, "tool_recall5": 0.56, "agent_recall2": 0.88},
    {"model": "Qwen2.5-7B-Instruct",   "layer": 28, "tool_recall5": 0.52, "agent_recall2": 0.87},
    {"model": "Mistral-7B-Instruct",   "layer": 0,  "tool_recall5": 0.09, "agent_recall2": 0.65},
    {"model": "Mistral-7B-Instruct",   "layer": 4,  "tool_recall5": 0.25, "agent_recall2": 0.75},
    {"model": "Mistral-7B-Instruct",   "layer": 8,  "tool_recall5": 0.38, "agent_recall2": 0.82},
    {"model": "Mistral-7B-Instruct",   "layer": 12, "tool_recall5": 0.52, "agent_recall2": 0.88},
    {"model": "Mistral-7B-Instruct",   "layer": 16, "tool_recall5": 0.57, "agent_recall2": 0.90},
    {"model": "Mistral-7B-Instruct",   "layer": 20, "tool_recall5": 0.53, "agent_recall2": 0.89},
    {"model": "Mistral-7B-Instruct",   "layer": 24, "tool_recall5": 0.49, "agent_recall2": 0.87},
    {"model": "Mistral-7B-Instruct",   "layer": 28, "tool_recall5": 0.45, "agent_recall2": 0.85},
]


def main():
    wandb.login(key=os.environ["WANDB_API_KEY"])

    for r in FAKE_RESULTS:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"TEST_{r['model']}_layer{r['layer']}",
            group=GROUP,
            tags=["test", "layer_sweep", r["model"]],
            config={
                "model": r["model"],
                "layer": r["layer"],
                "dataset": "mixed",
            },
            reinit=True,
        )

        wandb.log(
            {
                "tool_recall5": r["tool_recall5"],
                "agent_recall2": r["agent_recall2"],
            }
        )

        wandb.finish()
        print(f"Logged run: {run.name} "
              f"(tool_recall5={r['tool_recall5']}, agent_recall2={r['agent_recall2']})")

    print(
        "Done. Go to wandb.ai and check project ToolOptim "
        "under group layer_sweep_test"
    )


if __name__ == "__main__":
    main()
