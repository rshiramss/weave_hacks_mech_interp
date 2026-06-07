"""Three-arm routing benchmark: Weave Evaluation + wandb.Table (Stage 6, §7.2/§7.7).

Runs each arm over the published holdout with the routing scorers, then logs a
per-(arm x row) breakdown to a wandb.Table. Arms without a backend yet (probe)
are skipped with a warning so the baseline benchmark is never blocked.

Usage:
    python scripts/run_eval.py --dataset soc-routing-holdout --arms all
    python scripts/run_eval.py --arms incontext,rag
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(10_000)

from dotenv import load_dotenv  # noqa: E402

from eval.scorers import SCORERS, estimate_cost, routing_exact_match  # noqa: E402
from src.routers import DISPLAY_NAMES, ROUTERS  # noqa: E402
from src.weave_setup import init_weave  # noqa: E402

EVALUATION_NAME = "soc-routing-benchmark"
WANDB_PROJECT = "weavehacks-soc-probes"
TABLE_NAME = "soc-routing-eval-breakdown"
TABLE_COLUMNS = [
    "arm", "query_text", "gold_agent", "pred_agent", "gold_tool", "pred_tool",
    "agent_correct", "tool_correct", "routing_exact_match", "cost_usd", "latency_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three-arm routing benchmark.")
    parser.add_argument("--dataset", default="soc-routing-holdout")
    parser.add_argument("--arms", default="all",
                        help="'all' or csv of: incontext,rag,probe")
    return parser.parse_args()


def select_arms(arms_arg: str) -> list[str]:
    if arms_arg == "all":
        return ["incontext", "rag", "probe"]
    return [a.strip() for a in arms_arg.split(",") if a.strip()]


def run_weave_evals(dataset, arms: list[str]) -> list[str]:
    """Run a Weave Evaluation per arm; return the arms that actually ran."""
    import weave

    evaluation = weave.Evaluation(
        dataset=dataset, scorers=SCORERS, evaluation_name=EVALUATION_NAME
    )
    ran = []
    for arm in arms:
        try:
            asyncio.run(
                evaluation.evaluate(
                    ROUTERS[arm],
                    __weave={"display_name": DISPLAY_NAMES[arm]},
                )
            )
            ran.append(arm)
            print(f"[eval] Weave Evaluation done: {DISPLAY_NAMES[arm]}")
        except NotImplementedError as exc:
            print(f"[eval] SKIP {arm}: {exc}")
        except Exception as exc:  # noqa: BLE001 — keep other arms running
            print(f"[eval] ERROR {arm}: {exc}")
    return ran


def log_breakdown_table(dataset, arms: list[str]) -> None:
    """Per-(arm x row) breakdown to a wandb.Table (router calls are memoized)."""
    import wandb

    rows = list(dataset.rows)
    with wandb.init(project=WANDB_PROJECT, job_type="eval") as run:
        table = wandb.Table(columns=TABLE_COLUMNS)
        for arm in arms:
            for row in rows:
                try:
                    out = ROUTERS[arm](row["query_text"])
                except NotImplementedError:
                    continue
                agent_correct = out.get("agent_id") == row["agent_id"]
                tool_correct = out.get("tool_id") == row["tool_id"]
                table.add_data(
                    arm,
                    row["query_text"],
                    row["agent_id"],
                    out.get("agent_id"),
                    row["tool_id"],
                    out.get("tool_id"),
                    bool(agent_correct),
                    bool(tool_correct),
                    bool(routing_exact_match(row["agent_id"], row["tool_id"], out)),
                    estimate_cost(out.get("prompt_tokens"), out.get("completion_tokens")),
                    out.get("latency_ms"),
                )
        run.log({TABLE_NAME: table})
        print(f"[eval] logged '{TABLE_NAME}' to W&B project '{WANDB_PROJECT}'")


def main() -> int:
    load_dotenv()
    args = parse_args()
    arms = select_arms(args.arms)

    init_weave()
    import weave

    dataset = weave.ref(args.dataset).get()
    print(f"[eval] dataset '{args.dataset}' rows={len(list(dataset.rows))} arms={arms}\n")

    ran = run_weave_evals(dataset, arms)
    if ran:
        log_breakdown_table(dataset, ran)
    else:
        print("[eval] no arms ran — nothing to log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
