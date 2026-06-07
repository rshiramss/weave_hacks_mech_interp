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
import json
import sys
from collections import defaultdict
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


SUMMARY_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_summary.json"


def log_breakdown_table(dataset, arms: list[str]) -> None:
    """Per-(arm x row) breakdown to a wandb.Table + flat per-arm summary scalars.

    The flat `eval/{arm}/{metric}` summary keys are what the front-end /metrics
    endpoint reads back (run.summary scalars are reliably queryable via wandb.Api,
    unlike Weave Evaluation scorer aggregates). A local data/eval_summary.json is
    also written as an offline fallback. All string cells are str()-coerced —
    Weave-boxed strings otherwise break wandb.Table type inference.
    """
    import wandb

    rows = list(dataset.rows)
    agg: dict[str, dict[str, list]] = {arm: defaultdict(list) for arm in arms}

    with wandb.init(project=WANDB_PROJECT, job_type="eval") as run:
        table = wandb.Table(columns=TABLE_COLUMNS)
        for arm in arms:
            for row in rows:
                try:
                    out = ROUTERS[arm](row["query_text"])
                except NotImplementedError:
                    continue
                gold_agent, gold_tool = str(row["agent_id"]), str(row["tool_id"])
                pred_agent = str(out.get("agent_id") or "")
                pred_tool = str(out.get("tool_id") or "")
                agent_correct = pred_agent == gold_agent
                tool_correct = pred_tool == gold_tool
                rexact = bool(routing_exact_match(gold_agent, gold_tool, out))
                cost = estimate_cost(out.get("prompt_tokens"),
                                     out.get("completion_tokens"))
                latency = out.get("latency_ms")
                table.add_data(
                    arm, str(row["query_text"]), gold_agent, pred_agent,
                    gold_tool, pred_tool, agent_correct, tool_correct,
                    rexact, cost, latency,
                )

                bucket = agg[arm]
                bucket["routing_exact_match"].append(rexact)
                bucket["agent_top1"].append(agent_correct)
                bucket["tool_top1"].append(tool_correct)
                bucket["tool_in_shortlist"].append(
                    gold_tool in (out.get("tool_shortlist") or []))
                bucket["cost_usd"].append(cost)
                if latency is not None:
                    bucket["latency_ms"].append(latency)
                agent_cands = out.get("agent_candidates")
                tool_cands = out.get("tool_candidates")
                if agent_cands:
                    bucket["agent_recall_at_2"].append(gold_agent in agent_cands[:2])
                if tool_cands:
                    bucket["tool_recall_at_5"].append(gold_tool in tool_cands[:5])

        run.log({TABLE_NAME: table})

        summary: dict[str, dict] = {}
        for arm in arms:
            summary[arm] = {"n": len(rows)}
            run.summary[f"eval/{arm}/n"] = len(rows)
            for metric, values in agg[arm].items():
                if not values:
                    continue
                mean = sum(values) / len(values)
                run.summary[f"eval/{arm}/{metric}"] = mean
                summary[arm][metric] = mean
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[eval] logged '{TABLE_NAME}' + per-arm summary to '{WANDB_PROJECT}'")
        print(f"[eval] wrote {SUMMARY_PATH}")


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
