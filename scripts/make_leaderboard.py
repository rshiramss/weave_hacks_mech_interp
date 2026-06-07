"""Publish the routing leaderboard from the eval runs (Stage 7, §7.3).

Builds a Weave Leaderboard referencing the `soc-routing-benchmark` Evaluation
object, with columns for routing accuracy (maximize) and cost/latency (minimize).
Run AFTER scripts/run_eval.py has produced the arm runs.

    python scripts/make_leaderboard.py

If the programmatic API drifts, fall back to the UI steps printed at the end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.weave_setup import init_weave  # noqa: E402

EVALUATION_NAME = "soc-routing-benchmark"
DATASET_NAME = "soc-routing-holdout"
LEADERBOARD_NAME = "soc-routing-leaderboard"

# (scorer_name, summary_metric_path, minimize?)
COLUMNS = [
    ("routing_exact_match", "routing_exact_match.true_fraction", False),
    ("agent_recall_at_2", "agent_recall_at_2.true_fraction", False),
    ("tool_recall_at_5", "tool_recall_at_5.true_fraction", False),
    ("tool_in_shortlist", "tool_in_shortlist.true_fraction", False),
    ("estimated_cost", "estimated_cost.mean", True),
    ("latency_ms", "latency_ms.mean", True),
]


def _ui_steps():
    print("\n[leaderboard] UI fallback (§7.3):")
    print("  Weave → Evaluations → filter 'soc-routing-benchmark' → Visualize →")
    print("  Configure columns: routing_exact_match (higher), estimated_cost +")
    print("  latency_ms (lower) → Save view as 'soc-routing-leaderboard'.")


def main() -> int:
    load_dotenv()
    init_weave()

    import weave
    from weave.flow import leaderboard
    from weave.trace.ref_util import get_ref

    from eval.scorers import SCORERS

    try:
        dataset = weave.ref(DATASET_NAME).get()
        # Re-create the identical Evaluation object so its ref matches the runs.
        evaluation = weave.Evaluation(
            dataset=dataset, scorers=SCORERS, evaluation_name=EVALUATION_NAME
        )
        weave.publish(evaluation)
        eval_ref = get_ref(evaluation).uri()

        spec = leaderboard.Leaderboard(
            name=LEADERBOARD_NAME,
            description="SOC routing: probe vs RAG vs full-catalog in-context. "
                        "routing_exact_match higher = better; cost/latency lower = better.",
            columns=[
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=eval_ref,
                    scorer_name=scorer,
                    summary_metric_path=path,
                    should_minimize=minimize,
                )
                for scorer, path, minimize in COLUMNS
            ],
        )
        ref = weave.publish(spec)
        print(f"[leaderboard] published '{LEADERBOARD_NAME}'")
        print(f"[leaderboard] ref: {ref}")
        return 0
    except Exception as exc:  # noqa: BLE001 — programmatic API is version-sensitive
        print(f"[leaderboard] programmatic publish failed: {exc}")
        _ui_steps()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
